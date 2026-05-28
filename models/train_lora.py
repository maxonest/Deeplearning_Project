"""LoRA/QLoRA supervised fine-tuning for local causal language models.

This script intentionally uses Transformers Trainer + PEFT directly instead
of relying on TRL API details, making it easier to debug on Windows/Linux.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import format_sft_prompt, load_json_records, split_records  # noqa: E402


def parse_target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_tokenized_dataset(records: list[dict[str, Any]], tokenizer, max_length: int):
    from datasets import Dataset

    dataset = Dataset.from_list(records)

    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = format_sft_prompt(example)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def build_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = None
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    return AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        quantization_config=quantization_config,
    )


def train(args: argparse.Namespace) -> None:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    records = load_json_records(args.dataset_path)
    splits = split_records(records, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    if not splits["train"]:
        raise RuntimeError("No training records found.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(args)
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=parse_target_modules(args.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = build_tokenized_dataset(splits["train"], tokenizer, args.max_seq_length)
    eval_dataset = (
        build_tokenized_dataset(splits["validation"], tokenizer, args.max_seq_length)
        if splits["validation"]
        else None
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning.")
    parser.add_argument("--model_name_or_path", default="models/qwen/Qwen3.5-9B")
    parser.add_argument("--dataset_path", default="data/dataset.json")
    parser.add_argument("--output_dir", default="models/qwen/lora_adapter")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
