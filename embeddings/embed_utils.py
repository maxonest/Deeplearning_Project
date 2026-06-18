"""FAISS knowledge-base construction and retrieval.

Heavy dependencies (`faiss`, `sentence-transformers`) are imported lazily so
backend tests can run without a GPU model or an existing vector index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
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
INDEX_FORMAT_VERSION = 2
CORPUS_SUFFIXES = {".txt", ".md", ".csv", ".json", ".jsonl"}


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


def build_source_signature(
    input_paths: Iterable[str | Path],
    embedding_model: str,
    chunk_size: int,
    overlap: int,
) -> str:
    """Return a stable content fingerprint for all knowledge-base inputs."""

    digest = hashlib.sha256()
    digest.update(f"embedding_model={embedding_model}\n".encode())
    digest.update(f"chunk_size={chunk_size}\n".encode())
    digest.update(f"overlap={overlap}\n".encode())

    for source_index, raw_path in enumerate(input_paths):
        input_path = Path(raw_path)
        if input_path.is_file():
            files = [input_path]
            root = input_path.parent
        elif input_path.is_dir():
            files = sorted(
                path
                for path in input_path.rglob("*")
                if path.is_file() and path.suffix.lower() in CORPUS_SUFFIXES
            )
            root = input_path
        else:
            digest.update(f"missing:{source_index}:{input_path.name}\n".encode())
            continue

        for file_path in files:
            relative = file_path.relative_to(root).as_posix()
            digest.update(f"source:{source_index}:{relative}\n".encode("utf-8"))
            with file_path.open("rb") as source_file:
                for block in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


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
        faiss_threads: int = 1,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.device = device
        self.faiss_threads = max(1, faiss_threads)
        self._model = None
        self._index = None
        self._metadata: list[dict[str, Any]] | None = None
        self._lock = RLock()

    @property
    def exists(self) -> bool:
        return (self.index_dir / INDEX_FILE).exists() and (self.index_dir / METADATA_FILE).exists()

    def index_info(self) -> dict[str, int]:
        if self._index is None or self._metadata is None:
            self.load()
        assert self._index is not None
        assert self._metadata is not None
        return {
            "count": len(self._metadata),
            "dimension": int(self._index.d),
        }

    def _encoder(self):
        if self._model is None:
            SentenceTransformer = _load_sentence_transformer()
            self._model = SentenceTransformer(self.embedding_model, device=self.device)
        return self._model

    def _faiss(self):
        faiss = _load_faiss()
        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(self.faiss_threads)
        return faiss

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._encoder().encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return np.ascontiguousarray(embeddings, dtype="float32")

    def build(self, input_path: str | Path, chunk_size: int, overlap: int) -> int:
        return self.build_from_sources([input_path], chunk_size=chunk_size, overlap=overlap)

    def build_from_sources(
        self,
        input_paths: Iterable[str | Path],
        chunk_size: int,
        overlap: int,
    ) -> int:
        input_paths = [Path(path) for path in input_paths]
        chunks = build_chunks_from_sources(
            input_paths,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not chunks:
            raise RuntimeError("No text chunks found in the configured knowledge-base sources.")

        source_signature = build_source_signature(
            input_paths,
            embedding_model=self.embedding_model,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        with self._lock:
            embeddings = self.encode([chunk.text for chunk in chunks])
            self.save(embeddings, chunks, source_signature=source_signature)
        return len(chunks)

    def save(
        self,
        embeddings: np.ndarray,
        chunks: list[DocumentChunk],
        source_signature: str | None = None,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have the same length.")

        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        faiss = self._faiss()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        payload = {
            "format_version": INDEX_FORMAT_VERSION,
            "embedding_model": self.embedding_model,
            "dimension": int(embeddings.shape[1]),
            "source_signature": source_signature,
            "chunks": [asdict(chunk) for chunk in chunks],
        }
        index_path = self.index_dir / INDEX_FILE
        metadata_path = self.index_dir / METADATA_FILE
        temp_index_path = self.index_dir / f"{INDEX_FILE}.tmp"
        temp_metadata_path = self.index_dir / f"{METADATA_FILE}.tmp"
        try:
            faiss.write_index(index, str(temp_index_path))
            temp_metadata_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp_index_path, index_path)
            os.replace(temp_metadata_path, metadata_path)
        finally:
            temp_index_path.unlink(missing_ok=True)
            temp_metadata_path.unlink(missing_ok=True)
        self._index = index
        self._metadata = payload["chunks"]

    def load(self) -> None:
        with self._lock:
            if not self.exists:
                raise FileNotFoundError(f"FAISS index not found in {self.index_dir}")

            faiss = self._faiss()
            index = faiss.read_index(str(self.index_dir / INDEX_FILE))
            payload = json.loads((self.index_dir / METADATA_FILE).read_text(encoding="utf-8"))
            if isinstance(payload, list):
                raise RuntimeError(
                    "Legacy FAISS metadata has no embedding model or dimension. Rebuild the knowledge base."
                )
            if payload.get("format_version") != INDEX_FORMAT_VERSION:
                raise RuntimeError(
                    f"Unsupported FAISS metadata format: {payload.get('format_version')!r}. "
                    "Rebuild the knowledge base."
                )

            metadata = payload.get("chunks", [])
            stored_model = payload.get("embedding_model")
            stored_dimension = payload.get("dimension")
            if stored_model != self.embedding_model:
                raise RuntimeError(
                    "FAISS embedding model mismatch: "
                    f"index={stored_model!r}, runtime={self.embedding_model!r}. Rebuild the index."
                )
            if stored_dimension != index.d:
                raise RuntimeError(
                    f"FAISS dimension mismatch: metadata={stored_dimension}, index={index.d}."
                )
            if index.ntotal != len(metadata):
                raise RuntimeError(
                    f"FAISS item count mismatch: index={index.ntotal}, metadata={len(metadata)}."
                )
            self._index = index
            self._metadata = metadata

    def needs_rebuild(
        self,
        input_paths: Iterable[str | Path],
        chunk_size: int,
        overlap: int,
    ) -> bool:
        if not self.exists:
            return True
        try:
            payload = json.loads((self.index_dir / METADATA_FILE).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        if not isinstance(payload, dict):
            return True
        expected_signature = build_source_signature(
            input_paths,
            embedding_model=self.embedding_model,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        return (
            payload.get("format_version") != INDEX_FORMAT_VERSION
            or payload.get("embedding_model") != self.embedding_model
            or payload.get("source_signature") != expected_signature
        )

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        with self._lock:
            if self._index is None or self._metadata is None:
                self.load()

            assert self._index is not None
            assert self._metadata is not None

            query_vector = self.encode([query])
            if query_vector.ndim != 2 or query_vector.shape[1] != self._index.d:
                raise RuntimeError(
                    f"Query embedding dimension {query_vector.shape} does not match FAISS index dimension "
                    f"{self._index.d}."
                )
            effective_top_k = min(max(1, top_k), max(1, self._index.ntotal))
            scores, indices = self._index.search(query_vector, effective_top_k)

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
