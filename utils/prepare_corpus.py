"""Prepare raw corpus files for FAISS indexing.

This keeps original files under data/raw and writes cleaned text copies to
data/processed. JSON QA datasets can stay directly under data/processed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


TEXT_SUFFIXES = {".txt", ".md", ".csv"}


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prepare_raw_corpus(raw_dir: str | Path, processed_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for source in sorted(raw_dir.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = source.relative_to(raw_dir)
        target = processed_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(clean_text(source.read_text(encoding="utf-8", errors="ignore")), encoding="utf-8")
        written.append(target)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw corpus files into data/processed.")
    parser.add_argument("--raw_dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--processed_dir", default=str(PROJECT_ROOT / "data" / "processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = prepare_raw_corpus(args.raw_dir, args.processed_dir)
    for path in written:
        print(f"wrote {path}")
    print(f"Prepared {len(written)} files.")


if __name__ == "__main__":
    main()
