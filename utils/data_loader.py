"""Dataset utilities for SFT data and corpus preparation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_INSTRUCTION = "根据专业知识回答用户问题"


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON array or JSON Lines file and normalize records."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        raw_records = json.loads(text)
    else:
        raw_records = [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(raw_records, list):
        raise ValueError("Dataset must be a JSON array or JSON Lines file.")
    return [normalize_record(record) for record in raw_records]


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize common field names into instruction/input/output."""

    instruction = record.get("instruction") or DEFAULT_INSTRUCTION
    user_input = record.get("input") or record.get("question") or record.get("query") or ""
    output = record.get("output") or record.get("answer") or record.get("response") or ""
    metadata = record.get("metadata") or {}

    normalized = {
        "instruction": str(instruction).strip(),
        "input": str(user_input).strip(),
        "output": str(output).strip(),
        "metadata": metadata if isinstance(metadata, dict) else {"raw": metadata},
    }
    if not normalized["input"] or not normalized["output"]:
        raise ValueError(f"Record is missing input/output fields: {record}")
    return normalized


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically split records into train/validation/test."""

    if not records:
        return {"train": [], "validation": [], "test": []}
    if not 0 < train_ratio <= 1:
        raise ValueError("train_ratio must be in (0, 1].")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1).")
    if train_ratio + val_ratio > 1:
        raise ValueError("train_ratio + val_ratio must be <= 1.")

    items = records[:]
    random.Random(seed).shuffle(items)

    train_end = max(1, int(len(items) * train_ratio))
    val_end = train_end + int(len(items) * val_ratio)
    return {
        "train": items[:train_end],
        "validation": items[train_end:val_end],
        "test": items[val_end:],
    }


def save_splits(splits: dict[str, list[dict[str, Any]]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        output_path = output_dir / f"{name}.json"
        output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def to_dataset_dict(splits: dict[str, list[dict[str, Any]]]):
    """Convert splits to a Hugging Face DatasetDict lazily."""

    from datasets import Dataset, DatasetDict

    return DatasetDict({name: Dataset.from_list(records) for name, records in splits.items()})


def format_sft_prompt(record: dict[str, Any]) -> str:
    """Format one record in ChatML style for supervised fine-tuning."""

    instruction = record["instruction"].strip()
    user_input = record["input"].strip()
    output = record["output"].strip()
    user_content = f"{instruction}\n\n{user_input}" if instruction else user_input
    return (
        "<|im_start|>system\n你是一个专业、严谨的领域知识问答助手。<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def record_to_corpus_text(record: dict[str, Any]) -> str:
    """Convert a QA/dataset record into searchable corpus text."""

    normalized = normalize_record(record)
    source = normalized.get("metadata", {}).get("source", "")
    parts = [
        f"来源: {source}" if source else "",
        f"问题: {normalized['input']}",
        f"答案: {normalized['output']}",
    ]
    return "\n".join(part for part in parts if part).strip()


def read_corpus_files(input_path: str | Path) -> list[tuple[str, str]]:
    """Read text-like corpus files from a file or directory."""

    input_path = Path(input_path)
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*"))
    docs: list[tuple[str, str]] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix not in {".txt", ".md", ".csv", ".json", ".jsonl"}:
            continue
        if suffix in {".json", ".jsonl"}:
            try:
                records = load_json_records(file_path)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Failed to parse corpus dataset: {file_path}") from exc
            for index, record in enumerate(records):
                text = record_to_corpus_text(record)
                if text:
                    docs.append((f"{file_path}#{index}", text))
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append((str(file_path), text))
    return docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize and split QA dataset.")
    parser.add_argument("--input", default="data/dataset.json")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json_records(args.input)
    splits = split_records(records, seed=args.seed)
    save_splits(splits, args.output_dir)
    for name, values in splits.items():
        print(f"{name}: {len(values)} records")


if __name__ == "__main__":
    main()
