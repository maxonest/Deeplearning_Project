"""LoRA/QLoRA fine-tuning template for Qwen2.5, DeepSeek, or GLM models."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from utils.data_loader import format_sft_prompt, load_json_records, split_records, to_dataset_dict


def build_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_name_or_path: str, use_qlora: bool):
    quantization_config = None
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    return model


def build_lora_config(target_modules: list[str] | None = None) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules
        or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def train(args: argparse.Namespace) -> None:
    tokenizer = build_tokenizer(args.model_name_or_path)
    model = build_model(args.model_name_or_path, use_qlora=args.use_qlora)
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()

    records = load_json_records(args.dataset_path)
    dataset = to_dataset_dict(split_records(records, seed=args.seed))

    def formatting_func(example):
        return format_sft_prompt(example)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",
        save_total_limit=2,
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if len(dataset["validation"]) else None,
        formatting_func=formatting_func,
        args=training_args,
        max_seq_length=args.max_seq_length,
    )

    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_path", default="data/dataset.json")
    parser.add_argument("--output_dir", default="models/qwen/lora_adapter")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
