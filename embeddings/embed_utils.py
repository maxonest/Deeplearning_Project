"""Build and query a FAISS vector knowledge base from local text files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class DocumentChunk:
    id: int
    source: str
    text: str


def read_text_files(input_path: str | Path) -> list[tuple[str, str]]:
    """Read .txt/.md files from a file or directory."""

    input_path = Path(input_path)
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*"))
    docs: list[tuple[str, str]] = []
    for file_path in files:
        if file_path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append((str(file_path), text))
    return docs


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    """Split text into overlapping character chunks."""

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_chunks(input_path: str | Path, chunk_size: int, overlap: int) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for source, text in read_text_files(input_path):
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(DocumentChunk(id=len(chunks), source=source, text=chunk))
    return chunks


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype("float32")


def save_faiss_index(embeddings: np.ndarray, chunks: list[DocumentChunk], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "index.faiss"))

    metadata = [asdict(chunk) for chunk in chunks]
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_faiss_index(index_dir: str | Path) -> tuple[faiss.Index, list[dict]]:
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    return index, metadata


def search(
    query: str,
    index_dir: str | Path,
    model_name: str,
    top_k: int = 5,
) -> list[dict]:
    model = SentenceTransformer(model_name)
    index, metadata = load_faiss_index(index_dir)
    query_vector = encode_texts(model, [query], batch_size=1)
    scores, indices = index.search(query_vector, top_k)

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0], strict=False):
        if idx < 0:
            continue
        item = dict(metadata[idx])
        item["score"] = float(score)
        results.append(item)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from local corpus.")
    parser.add_argument("--input", default="data/processed")
    parser.add_argument("--output", default="embeddings/faiss_index")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--chunk_size", type=int, default=600)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    chunks = build_chunks(args.input, chunk_size=args.chunk_size, overlap=args.overlap)
    if not chunks:
        raise RuntimeError(f"No text chunks found under {args.input}")

    model = SentenceTransformer(args.model)
    embeddings = encode_texts(model, [chunk.text for chunk in chunks], batch_size=args.batch_size)
    save_faiss_index(embeddings, chunks, args.output)
    print(f"Saved {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
