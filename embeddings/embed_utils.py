"""FAISS knowledge-base construction and retrieval.

Heavy dependencies (`faiss`, `sentence-transformers`) are imported lazily so
backend tests can run without a GPU model or an existing vector index.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import read_corpus_files


INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "embeddings" / "faiss_index"
DEFAULT_PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class DocumentChunk:
    id: int
    source: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    id: int
    source: str
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss is not installed. Install faiss-cpu or use conda-forge faiss-cpu.") from exc
    return faiss


def _load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed.") from exc
    return SentenceTransformer


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def build_chunks(input_path: str | Path, chunk_size: int, overlap: int) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for source, text in read_corpus_files(input_path):
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(DocumentChunk(id=len(chunks), source=source, text=chunk))
    return chunks


def build_chunks_from_sources(
    input_paths: Iterable[str | Path],
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for input_path in input_paths:
        for source, text in read_corpus_files(input_path):
            source_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for chunk in source_chunks:
                chunks.append(
                    DocumentChunk(
                        id=len(chunks),
                        source=source,
                        text=chunk,
                    )
                )
    return chunks


class FaissKnowledgeBase:
    """A small local vector store backed by FAISS."""

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._index = None
        self._metadata: list[dict[str, Any]] | None = None

    @property
    def exists(self) -> bool:
        return (self.index_dir / INDEX_FILE).exists() and (self.index_dir / METADATA_FILE).exists()

    def _encoder(self):
        if self._model is None:
            SentenceTransformer = _load_sentence_transformer()
            self._model = SentenceTransformer(self.embedding_model, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._encoder().encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return embeddings.astype("float32")

    def build(self, input_path: str | Path, chunk_size: int, overlap: int) -> int:
        return self.build_from_sources([input_path], chunk_size=chunk_size, overlap=overlap)

    def build_from_sources(
        self,
        input_paths: Iterable[str | Path],
        chunk_size: int,
        overlap: int,
    ) -> int:
        chunks = build_chunks_from_sources(
            input_paths,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not chunks:
            raise RuntimeError("No text chunks found in the configured knowledge-base sources.")

        embeddings = self.encode([chunk.text for chunk in chunks])
        self.save(embeddings, chunks)
        return len(chunks)

    def save(self, embeddings: np.ndarray, chunks: list[DocumentChunk]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have the same length.")

        faiss = _load_faiss()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(self.index_dir / INDEX_FILE))

        payload = {
            "embedding_model": self.embedding_model,
            "chunks": [asdict(chunk) for chunk in chunks],
        }
        (self.index_dir / METADATA_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._index = index
        self._metadata = payload["chunks"]

    def load(self) -> None:
        if not self.exists:
            raise FileNotFoundError(f"FAISS index not found in {self.index_dir}")

        faiss = _load_faiss()
        self._index = faiss.read_index(str(self.index_dir / INDEX_FILE))
        payload = json.loads((self.index_dir / METADATA_FILE).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            self._metadata = payload
        else:
            self._metadata = payload.get("chunks", [])

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        if self._index is None or self._metadata is None:
            self.load()

        assert self._index is not None
        assert self._metadata is not None

        query_vector = self.encode([query])
        scores, indices = self._index.search(query_vector, top_k)

        hits: list[dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0 or index >= len(self._metadata):
                continue
            item = dict(self._metadata[index])
            item["score"] = float(score)
            hits.append(item)
        return hits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query a FAISS knowledge base.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--input", default=str(DEFAULT_PROCESSED_DATA_DIR))
    build_parser.add_argument("--output", default=str(DEFAULT_INDEX_DIR))
    build_parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    build_parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    build_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("query")
    query_parser.add_argument("--index", default=str(DEFAULT_INDEX_DIR))
    query_parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    query_parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        kb = FaissKnowledgeBase(index_dir=args.output, embedding_model=args.model)
        count = kb.build(args.input, chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"Saved {count} chunks to {args.output}")
    elif args.command == "query":
        kb = FaissKnowledgeBase(index_dir=args.index, embedding_model=args.model)
        print(json.dumps(kb.search(args.query, top_k=args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
