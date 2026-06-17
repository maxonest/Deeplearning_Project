"""LoRA/QLoRA supervised fine-tuning for local causal language models.

The training loop is intentionally implemented without Hugging Face Trainer.
On some Windows environments, importing Trainer pulls in datasets/pyarrow and
can trigger native access violations before Python can raise a normal error.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import load_json_records, split_records  # noqa: E402


LOG_DIR = PROJECT_ROOT / "logs"
DEBUG_LOG_PATH = LOG_DIR / "train_lora_debug.log"


TRAINING_DEFAULTS = {
    "model_name_or_path": "models/qwen/Qwen3.5-9B",
    "dataset_path": "data/finetune/sft_dataset_clean.json",
    "output_dir": "models/qwen/lora_adapter",
    "local_files_only": True,
    "epochs": 1.0,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_grad_norm": 1.0,
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
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def info(message: str) -> None:
    print(message, flush=True)
    log_stage(message)


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


def record_to_messages(record: dict[str, Any], include_answer: bool) -> list[dict[str, str]]:
    instruction = record["instruction"].strip()
    user_input = record["input"].strip()
    output = record["output"].strip()
    user_content = f"{instruction}\n\n{user_input}" if instruction else user_input
    messages = [
        {"role": "system", "content": "你是一个专业、严谨的运动健康领域问答助手。"},
        {"role": "user", "content": user_content},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": output})
    return messages


def encode_record(record: dict[str, Any], tokenizer, max_length: int) -> dict[str, list[int]]:
    if getattr(tokenizer, "chat_template", None):
        prompt_ids = tokenizer.apply_chat_template(
            record_to_messages(record, include_answer=False),
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            record_to_messages(record, include_answer=True),
            tokenize=True,
            add_generation_prompt=False,
        )
    else:
        prompt, answer = format_prompt_parts(record)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + tokenizer(answer, add_special_tokens=False)["input_ids"]

    if len(full_ids) <= len(prompt_ids):
        prompt, answer = format_prompt_parts(record)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = full_ids[:max_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels[: len(input_ids)],
    }


def tokenize_records(records: list[dict[str, Any]], tokenizer, max_length: int) -> list[dict[str, list[int]]]:
    tokenized_records = []
    for record in records:
        tokenized_records.append(encode_record(record, tokenizer, max_length))
    return tokenized_records


def collate_batch(features: list[dict[str, list[int]]], pad_token_id: int):
    import torch

    max_length = max(len(item["input_ids"]) for item in features)
    input_ids, attention_mask, labels = [], [], []
    for item in features:
        pad_length = max_length - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_length)
        attention_mask.append(item["attention_mask"] + [0] * pad_length)
        labels.append(item["labels"] + [-100] * pad_length)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def preview_dataset(records: list[dict[str, Any]], tokenizer, max_length: int) -> None:
    if not records:
        return
    tokenized = encode_record(records[0], tokenizer, max_length)
    trainable_tokens = sum(1 for label in tokenized["labels"] if label != -100)
    preview_text = tokenizer.decode(tokenized["input_ids"][:max_length], skip_special_tokens=False)
    print("=" * 80)
    print("Dataset preview")
    print("=" * 80)
    print(preview_text[:1200])
    print(
        f"tokens: {len(tokenized['input_ids'])}, "
        f"trainable_tokens: {trainable_tokens}, max_seq_length: {max_length}"
    )
    print("=" * 80)


def validate_training_environment(args: argparse.Namespace) -> None:
    import torch

    log_stage("Validating training environment.")
    if args.bf16 and args.fp16:
        raise RuntimeError("bf16 and fp16 cannot both be enabled. Use --no_bf16 --fp16 for fp16.")
    if not Path(args.model_name_or_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {Path(args.model_name_or_path).resolve()}")
    if not Path(args.dataset_path).exists():
        raise FileNotFoundError(f"Dataset path does not exist: {Path(args.dataset_path).resolve()}")

    if not torch.cuda.is_available():
        message = "CUDA is not available in this Python environment."
        if args.require_cuda:
            raise RuntimeError(message + " Check your conda env/PyTorch CUDA build, or pass --allow_cpu for tiny tests.")
        info("WARNING: " + message)
        return

    device_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    info(f"CUDA: {device_name}, total_memory={total_gb:.2f}G")

    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 is not supported. Try: python models/train_lora.py --no_bf16 --fp16")
    if args.use_qlora:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("QLoRA requires bitsandbytes. Run LoRA without --use_qlora first.") from exc


def get_model_dtype(args: argparse.Namespace):
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def build_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    log_stage("Loading model.")
    quantization_config = None
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=get_model_dtype(args),
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        quantization_config=quantization_config,
    )
    log_stage("Model loaded.")
    return model


def move_batch_to_device(batch: dict[str, Any], device):
    return {key: value.to(device) for key, value in batch.items()}


def get_primary_device(model):
    return next(parameter.device for parameter in model.parameters() if parameter is not None)


def init_swanlab(args: argparse.Namespace):
    if not args.use_swanlab:
        return None
    try:
        import swanlab

        run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_run_name,
            config=vars(args),
        )
        info(f"SwanLab: project={args.swanlab_project}, run={args.swanlab_run_name}")
        return swanlab
    except ImportError:
        info("SwanLab not installed. Continuing without SwanLab.")
        return None
    except Exception as exc:
        info(f"SwanLab disabled: {exc}")
        return None


def evaluate(model, eval_loader, device) -> float | None:
    import torch

    if eval_loader is None:
        return None
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            losses.append(outputs.loss.detach().float().item())
    model.train()
    return sum(losses) / len(losses) if losses else None


def save_adapter(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    info("LoRA fine-tuning started.")
    info(
        "Config: "
        f"model={args.model_name_or_path}, dataset={args.dataset_path}, "
        f"epochs={args.epochs}, batch={args.per_device_train_batch_size}, "
        f"grad_accum={args.gradient_accumulation_steps}, max_seq={args.max_seq_length}"
    )

    records = load_json_records(args.dataset_path)
    if args.max_samples:
        records = records[: args.max_samples]
    splits = split_records(records, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    if not splits["train"]:
        raise RuntimeError("No training records found.")
    info(f"Dataset: train={len(splits['train'])}, validation={len(splits['validation'])}, test={len(splits['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    preview_dataset(splits["train"], tokenizer, args.max_seq_length)
    if args.dry_run:
        info("Dry run finished. No training started.")
        return

    validate_training_environment(args)

    info("Loading base model...")
    model = build_model(args)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
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
    model.train()

    train_features = tokenize_records(splits["train"], tokenizer, args.max_seq_length)
    eval_features = tokenize_records(splits["validation"], tokenizer, args.max_seq_length) if splits["validation"] else []
    if not train_features:
        raise RuntimeError("Tokenized train dataset is empty.")

    train_loader = DataLoader(
        train_features,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    eval_loader = (
        DataLoader(eval_features, batch_size=1, shuffle=False, collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id))
        if eval_features
        else None
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    total_steps = max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    info(f"Training steps: {total_steps}")

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    swanlab = init_swanlab(args)
    device = get_primary_device(model)

    global_step = 0
    running_loss = 0.0
    running_loss_count = 0
    optimizer.zero_grad(set_to_none=True)
    info(f"Training loop started. warmup_steps={warmup_steps}")

    for epoch in range(math.ceil(args.epochs)):
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += raw_loss.detach().float().item()
            running_loss_count += 1

            if batch_index % args.gradient_accumulation_steps != 0 and batch_index != len(train_loader):
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            grad_norm_value = float(grad_norm.detach().float().item())
            if global_step == 0 and (not math.isfinite(grad_norm_value) or grad_norm_value == 0.0):
                raise RuntimeError(
                    f"Invalid LoRA gradient norm at first optimizer step: {grad_norm_value}. "
                    "The adapter may not be connected to the loss."
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.logging_steps == 0 or global_step == 1:
                current_loss = running_loss / max(1, running_loss_count)
                running_loss = 0.0
                running_loss_count = 0
                lr = scheduler.get_last_lr()[0]
                message = (
                    f"step {global_step}/{total_steps} | loss={current_loss:.4f} "
                    f"| lr={lr:.3e} | grad_norm={grad_norm_value:.4f}"
                )
                if torch.cuda.is_available():
                    used_gb = torch.cuda.memory_allocated() / 1024**3
                    message += f" | cuda_mem={used_gb:.2f}G"
                info(message)
                if swanlab is not None:
                    swanlab.log(
                        {
                            "train/loss": current_loss,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm_value,
                        },
                        step=global_step,
                    )

            if eval_loader is not None and global_step % args.eval_steps == 0:
                eval_loss = evaluate(model, eval_loader, device)
                if eval_loss is not None:
                    info(f"eval step {global_step}/{total_steps} | loss={eval_loss:.4f}")
                    if swanlab is not None:
                        swanlab.log({"eval/loss": eval_loss}, step=global_step)

            if global_step % args.save_steps == 0:
                checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                save_adapter(model, tokenizer, checkpoint_dir)
                info(f"Checkpoint saved: {checkpoint_dir}")

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    if global_step <= 0:
        raise RuntimeError("Training finished with global_step=0. No optimizer step was executed.")

    output_dir = Path(args.output_dir)
    save_adapter(model, tokenizer, output_dir)
    info(f"Training finished. LoRA adapter saved to: {output_dir.resolve()}")
    if swanlab is not None:
        try:
            swanlab.finish()
        except Exception:
            pass


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
    parser.add_argument("--warmup_ratio", type=float, default=TRAINING_DEFAULTS["warmup_ratio"])
    parser.add_argument("--per_device_train_batch_size", type=int, default=TRAINING_DEFAULTS["per_device_train_batch_size"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=TRAINING_DEFAULTS["gradient_accumulation_steps"])
    parser.add_argument("--max_grad_norm", type=float, default=TRAINING_DEFAULTS["max_grad_norm"])
    parser.add_argument("--max_seq_length", type=int, default=TRAINING_DEFAULTS["max_seq_length"])
    parser.add_argument("--max_samples", type=int, default=0, help="Use first N samples for smoke testing.")
    parser.add_argument("--train_ratio", type=float, default=TRAINING_DEFAULTS["train_ratio"])
    parser.add_argument("--val_ratio", type=float, default=TRAINING_DEFAULTS["val_ratio"])
    parser.add_argument("--seed", type=int, default=TRAINING_DEFAULTS["seed"])
    parser.add_argument("--lora_r", type=int, default=TRAINING_DEFAULTS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=TRAINING_DEFAULTS["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=TRAINING_DEFAULTS["lora_dropout"])
    parser.add_argument("--target_modules", default=TRAINING_DEFAULTS["target_modules"])
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
        train(parse_args())
    except Exception:
        error_text = traceback.format_exc()
        log_stage("Training failed with an explicit Python exception.")
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write("\n" + error_text + "\n")
        print("\nTraining failed:\n", file=sys.stderr)
        print(error_text, file=sys.stderr)
        sys.exit(1)
