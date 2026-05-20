"""Dataset loading, normalization, and train/validation/test splitting."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON array or JSON Lines records."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(records, list):
        raise ValueError("Dataset must be a JSON array or JSON Lines file.")
    return [normalize_record(record) for record in records]


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize common QA fields into instruction/input/output."""

    instruction = record.get("instruction") or "根据专业知识回答用户问题"
    question = record.get("input") or record.get("question") or record.get("query") or ""
    answer = record.get("output") or record.get("answer") or record.get("response") or ""
    metadata = record.get("metadata") or {}

    return {
        "instruction": str(instruction).strip(),
        "input": str(question).strip(),
        "output": str(answer).strip(),
        "metadata": metadata,
    }


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split records into train, validation, and test subsets."""

    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be in (0, 1).")

    items = records[:]
    random.Random(seed).shuffle(items)
    train_end = int(len(items) * train_ratio)
    val_end = train_end + int(len(items) * val_ratio)

    return {
        "train": items[:train_end],
        "validation": items[train_end:val_end],
        "test": items[val_end:],
    }


def to_dataset_dict(splits: dict[str, list[dict[str, Any]]]) -> DatasetDict:
    return DatasetDict({name: Dataset.from_list(records) for name, records in splits.items()})


def format_sft_prompt(record: dict[str, Any]) -> str:
    """Format one record for supervised fine-tuning."""

    return (
        "<|im_start|>system\n你是一个专业、严谨的领域知识问答助手。<|im_end|>\n"
        f"<|im_start|>user\n{record['instruction']}\n\n{record['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{record['output']}<|im_end|>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/dataset.json")
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    records = load_json_records(args.input)
    splits = split_records(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_records_ in splits.items():
        output_path = output_dir / f"{split_name}.json"
        output_path.write_text(
            json.dumps(split_records_, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {len(split_records_)} records to {output_path}")


if __name__ == "__main__":
    main()
