"""LoRA/QLoRA supervised fine-tuning for local causal language models.

This script intentionally uses Transformers Trainer + PEFT directly instead
of relying on TRL API details, making it easier to debug on Windows/Linux.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import load_json_records, split_records  # noqa: E402


LOG_DIR = PROJECT_ROOT / "logs"
DEBUG_LOG_PATH = LOG_DIR / "train_lora_debug.log"


TRAINING_DEFAULTS = {
    # Windows deployment default:
    # D:/lyx/Deeplearning_Project/models/qwen/Qwen3.5-9B
    "model_name_or_path": "models/qwen/Qwen3.5-9B",
    "dataset_path": "data/finetune/sft_dataset_clean.json",
    "output_dir": "models/qwen/lora_adapter",
    "local_files_only": True,
    # Conservative LoRA defaults for a single RTX 4090/4090D.
    "epochs": 1.0,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 1024,
    "train_ratio": 0.9,
    "val_ratio": 0.1,
    "seed": 42,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "logging_steps": 10,
    "save_steps": 200,
    "eval_steps": 200,
    "gradient_checkpointing": True,
    "bf16": True,
    "fp16": False,
    "require_cuda": True,
    "use_swanlab": True,
    "swanlab_project": "local-domain-qa-lora",
    "swanlab_run_name": "qwen3.5-9b-lora",
}


def log_stage(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def enable_fault_diagnostics() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    faulthandler.enable(file=sys.stderr, all_threads=True)
    log_stage("Fault diagnostics enabled on terminal stderr.")
    log_stage("Native debug env: CUDA_LAUNCH_BLOCKING=1, TORCH_SHOW_CPP_STACKTRACES=1.")


def parse_target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def format_prompt_parts(record: dict[str, Any]) -> tuple[str, str]:
    instruction = record["instruction"].strip()
    user_input = record["input"].strip()
    output = record["output"].strip()
    user_content = f"{instruction}\n\n{user_input}" if instruction else user_input
    prompt = (
        "<|im_start|>system\n你是一个专业、严谨的运动健康领域问答助手。<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    answer = f"{output}<|im_end|>"
    return prompt, answer


def build_tokenized_dataset(records: list[dict[str, Any]], tokenizer, max_length: int):
    from datasets import Dataset

    dataset = Dataset.from_list(records)

    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        prompt, answer = format_prompt_parts(example)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + answer_ids)[:max_length]
        labels = [-100] * min(len(prompt_ids), len(input_ids))
        labels += input_ids[len(labels) :]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels[: len(input_ids)],
        }

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def preview_dataset(records: list[dict[str, Any]], tokenizer, max_length: int) -> None:
    if not records:
        return
    prompt, answer = format_prompt_parts(records[0])
    tokenized = tokenizer(
        prompt + answer,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    print("=" * 80)
    print("Dataset preview")
    print("=" * 80)
    print((prompt + answer)[:1200])
    print(f"tokens: {len(tokenized['input_ids'])}, max_seq_length: {max_length}")
    print("=" * 80)


def build_training_arguments(args: argparse.Namespace, has_eval_dataset: bool):
    log_stage("Building TrainingArguments.")
    from transformers import TrainingArguments

    kwargs = dict(
        output_dir=args.output_dir,
        run_name=args.swanlab_run_name,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
        dataloader_num_workers=0,
        disable_tqdm=False,
    )
    strategy = "steps" if has_eval_dataset else "no"
    try:
        return TrainingArguments(eval_strategy=strategy, **kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy=strategy, **kwargs)


def build_model(args: argparse.Namespace):
    log_stage("Importing torch and Transformers model classes.")
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    log_stage("Model class imports finished.")
    quantization_config = None
    if args.use_qlora:
        log_stage("Building QLoRA BitsAndBytesConfig.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    log_stage(f"Loading base model from: {Path(args.model_name_or_path).resolve()}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        quantization_config=quantization_config,
    )
    log_stage("Base model loaded successfully.")
    return model


def validate_training_environment(args: argparse.Namespace) -> None:
    log_stage("Validating training environment.")
    import torch

    if args.bf16 and args.fp16:
        raise RuntimeError("bf16 and fp16 cannot be enabled at the same time. Use --no_bf16 --fp16 for fp16.")

    if not Path(args.model_name_or_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {Path(args.model_name_or_path).resolve()}")

    if not Path(args.dataset_path).exists():
        raise FileNotFoundError(f"Dataset path does not exist: {Path(args.dataset_path).resolve()}")

    if not torch.cuda.is_available():
        message = (
            "CUDA is not available in this Python environment. "
            "Check that you activated the right conda env and installed the CUDA build of PyTorch."
        )
        if args.require_cuda:
            raise RuntimeError(message + " To force CPU training, pass --allow_cpu.")
        print("WARNING: " + message, flush=True)
        return

    device_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log_stage(f"CUDA ready: {device_name}, total_memory={total_gb:.2f}G")

    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 is not supported by this GPU/PyTorch build. Try: python models/train_lora.py --no_bf16 --fp16")

    if args.use_qlora:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "QLoRA requires bitsandbytes, but it is not installed. "
                "Use LoRA without --use_qlora first, or install a Windows-compatible bitsandbytes build."
            ) from exc


def build_monitoring_callbacks(args: argparse.Namespace):
    log_stage("Building monitoring callbacks.")
    from transformers import TrainerCallback

    class ConsoleProgressCallback(TrainerCallback):
        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if not state.is_local_process_zero or not logs:
                return

            parts = [f"step={state.global_step}/{state.max_steps}"]
            for key in ("loss", "eval_loss", "learning_rate", "grad_norm"):
                if key in logs:
                    value = logs[key]
                    parts.append(f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}")

            try:
                import torch

                if torch.cuda.is_available():
                    used_gb = torch.cuda.memory_allocated() / 1024**3
                    reserved_gb = torch.cuda.memory_reserved() / 1024**3
                    parts.append(f"cuda_mem={used_gb:.2f}G/{reserved_gb:.2f}G")
            except Exception:
                pass

            print("[train] " + " | ".join(parts), flush=True)

    callbacks = [ConsoleProgressCallback()]
    if not args.use_swanlab:
        return callbacks

    try:
        import swanlab
        from swanlab.integration.transformers import SwanLabCallback
    except ImportError:
        print(
            "SwanLab is not installed. Install it with: python -m pip install swanlab. "
            "Training will continue with console progress only.",
            flush=True,
        )
        return callbacks

    config = vars(args).copy()
    try:
        callbacks.append(
            SwanLabCallback(
                project=args.swanlab_project,
                experiment_name=args.swanlab_run_name,
                config=config,
            )
        )
        print(f"SwanLab enabled: project={args.swanlab_project}, run={args.swanlab_run_name}", flush=True)
    except TypeError:
        try:
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_run_name,
                config=config,
            )
            callbacks.append(SwanLabCallback())
            print(f"SwanLab enabled: project={args.swanlab_project}, run={args.swanlab_run_name}", flush=True)
        except Exception as exc:
            print(f"SwanLab initialization failed: {exc}. Training will continue with console progress only.")
    except Exception as exc:
        print(f"SwanLab initialization failed: {exc}. Training will continue with console progress only.")

    return callbacks


def train(args: argparse.Namespace) -> None:
    log_stage("Importing tokenizer.")
    from transformers import AutoTokenizer

    print("Effective training config:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}", flush=True)

    log_stage("Loading JSON records.")
    records = load_json_records(args.dataset_path)
    if args.max_samples:
        records = records[: args.max_samples]
    log_stage(f"Loaded records: {len(records)} from {args.dataset_path}")
    log_stage("Splitting dataset.")
    splits = split_records(records, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    log_stage(
        f"Dataset split: train={len(splits['train'])}, validation={len(splits['validation'])}, test={len(splits['test'])}",
    )
    if not splits["train"]:
        raise RuntimeError("No training records found.")

    log_stage("Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    log_stage("Tokenizer loaded successfully.")

    preview_dataset(splits["train"], tokenizer, args.max_seq_length)
    if args.dry_run:
        print("Dry run finished. No training started.")
        return

    validate_training_environment(args)

    log_stage("Importing PEFT and Trainer classes.")
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForSeq2Seq, Trainer
    log_stage("PEFT and Trainer imports finished.")

    model = build_model(args)
    if args.gradient_checkpointing and hasattr(model, "config"):
        log_stage("Disabling model.config.use_cache for gradient checkpointing.")
        model.config.use_cache = False
    if args.use_qlora:
        log_stage("Preparing model for k-bit training.")
        model = prepare_model_for_kbit_training(model)

    log_stage("Applying LoRA adapter.")
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
    log_stage("LoRA adapter applied.")

    log_stage("Tokenizing train dataset.")
    train_dataset = build_tokenized_dataset(splits["train"], tokenizer, args.max_seq_length)
    log_stage("Tokenizing validation dataset.")
    eval_dataset = (
        build_tokenized_dataset(splits["validation"], tokenizer, args.max_seq_length)
        if splits["validation"]
        else None
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Tokenized train dataset is empty. Check dataset_path and max_seq_length.")

    estimated_steps = (
        len(train_dataset)
        * args.epochs
        / max(1, args.per_device_train_batch_size)
        / max(1, args.gradient_accumulation_steps)
    )
    log_stage(f"Tokenized train samples: {len(train_dataset)}")
    log_stage(f"Estimated optimizer steps: {estimated_steps:.1f}")
    if estimated_steps < 1:
        raise RuntimeError(
            "Estimated optimizer steps is less than 1. "
            "Increase epochs/max_samples or reduce gradient_accumulation_steps."
        )

    training_args = build_training_arguments(args, has_eval_dataset=eval_dataset is not None)
    callbacks = build_monitoring_callbacks(args)

    log_stage("Initializing Trainer.")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=callbacks,
    )
    log_stage("Training started.")
    train_result = trainer.train()
    if trainer.state.global_step <= 0:
        raise RuntimeError(
            "Training finished with global_step=0. No optimizer step was executed. "
            "Check dataset size, epochs, batch size, and gradient_accumulation_steps."
        )
    log_stage(f"Training finished: global_step={trainer.state.global_step}")
    log_stage(f"Training metrics: {train_result.metrics}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_stage("Saving LoRA adapter and tokenizer.")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log_stage(f"LoRA adapter saved to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning.")
    parser.add_argument("--model_name_or_path", default=TRAINING_DEFAULTS["model_name_or_path"])
    parser.add_argument("--dataset_path", default=TRAINING_DEFAULTS["dataset_path"])
    parser.add_argument("--output_dir", default=TRAINING_DEFAULTS["output_dir"])
    parser.add_argument(
        "--allow_remote_download",
        dest="local_files_only",
        action="store_false",
        default=TRAINING_DEFAULTS["local_files_only"],
        help="Allow Transformers to download missing model/tokenizer files.",
    )
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--epochs", type=float, default=TRAINING_DEFAULTS["epochs"])
    parser.add_argument("--learning_rate", type=float, default=TRAINING_DEFAULTS["learning_rate"])
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=TRAINING_DEFAULTS["per_device_train_batch_size"],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=TRAINING_DEFAULTS["gradient_accumulation_steps"],
    )
    parser.add_argument("--max_seq_length", type=int, default=TRAINING_DEFAULTS["max_seq_length"])
    parser.add_argument("--max_samples", type=int, default=0, help="Use first N samples for smoke testing.")
    parser.add_argument("--train_ratio", type=float, default=TRAINING_DEFAULTS["train_ratio"])
    parser.add_argument("--val_ratio", type=float, default=TRAINING_DEFAULTS["val_ratio"])
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULTS["seed"])
    parser.add_argument("--lora_r", type=int, default=TRAINING_DEFAULTS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=TRAINING_DEFAULTS["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=TRAINING_DEFAULTS["lora_dropout"])
    parser.add_argument(
        "--target_modules",
        default=TRAINING_DEFAULTS["target_modules"],
    )
    parser.add_argument("--logging_steps", type=int, default=TRAINING_DEFAULTS["logging_steps"])
    parser.add_argument("--save_steps", type=int, default=TRAINING_DEFAULTS["save_steps"])
    parser.add_argument("--eval_steps", type=int, default=TRAINING_DEFAULTS["eval_steps"])
    parser.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        default=TRAINING_DEFAULTS["gradient_checkpointing"],
    )
    parser.add_argument("--no_bf16", dest="bf16", action="store_false", default=TRAINING_DEFAULTS["bf16"])
    parser.add_argument("--fp16", action="store_true", default=TRAINING_DEFAULTS["fp16"])
    parser.add_argument(
        "--allow_cpu",
        dest="require_cuda",
        action="store_false",
        default=TRAINING_DEFAULTS["require_cuda"],
        help="Allow training to continue without CUDA. This is only useful for tiny debugging runs.",
    )
    parser.add_argument(
        "--no_swanlab",
        dest="use_swanlab",
        action="store_false",
        default=TRAINING_DEFAULTS["use_swanlab"],
        help="Disable SwanLab experiment tracking.",
    )
    parser.add_argument("--swanlab_project", default=TRAINING_DEFAULTS["swanlab_project"])
    parser.add_argument("--swanlab_run_name", default=TRAINING_DEFAULTS["swanlab_run_name"])
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        try:
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except AttributeError:
            pass
        enable_fault_diagnostics()
        log_stage("Training process started.")
        train(parse_args())
        log_stage("Training process exited normally.")
    except Exception:
        error_text = traceback.format_exc()
        log_stage("Training failed with an explicit Python exception.")
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write("\n" + error_text + "\n")
        print("\nTraining failed with an explicit error:\n", file=sys.stderr)
        print(error_text, file=sys.stderr)
        sys.exit(1)
