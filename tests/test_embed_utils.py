import json

import numpy as np

import embeddings.embed_utils as embed_utils
from embeddings.embed_utils import (
    DocumentChunk,
    FaissKnowledgeBase,
    build_chunks_from_sources,
    build_source_signature,
)


def test_build_chunks_from_sources_includes_sft_dataset(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "manual.txt").write_text("普通专业资料", encoding="utf-8")
    finetune_path = tmp_path / "sft_dataset_clean.json"
    finetune_path.write_text(
        json.dumps(
            [
                {
                    "instruction": "回答运动问题",
                    "input": "什么是体适能？",
                    "output": "体适能是完成体力活动所需的一组能力。",
                    "metadata": {"source": "ACSM指南"},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    chunks = build_chunks_from_sources(
        [processed_dir, finetune_path],
        chunk_size=600,
        overlap=80,
    )

    assert len(chunks) == 2
    assert chunks[0].id == 0
    assert chunks[1].id == 1
    assert "什么是体适能" in chunks[1].text
    assert "体适能是完成体力活动" in chunks[1].text


def test_source_signature_changes_with_corpus_content(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "manual.txt"
    source.write_text("第一版", encoding="utf-8")

    first = build_source_signature(
        [corpus],
        embedding_model="test-model",
        chunk_size=600,
        overlap=80,
    )
    source.write_text("第二版", encoding="utf-8")
    second = build_source_signature(
        [corpus],
        embedding_model="test-model",
        chunk_size=600,
        overlap=80,
    )

    assert first != second


def test_encode_reports_batch_progress():
    class FakeEncoder:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 3), dtype="float32")

    knowledge_base = FaissKnowledgeBase(batch_size=2)
    knowledge_base._model = FakeEncoder()
    progress = []

    embeddings = knowledge_base.encode(
        ["a", "b", "c", "d", "e"],
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert embeddings.shape == (5, 3)
    assert progress == [(0, 5), (2, 5), (4, 5), (5, 5)]


def test_load_falls_back_to_previous_valid_index(monkeypatch, tmp_path):
    class FakeIndex:
        def __init__(self, dimension):
            self.d = dimension
            self.ntotal = 0

        def add(self, embeddings):
            self.ntotal = len(embeddings)

    class FakeFaiss:
        @staticmethod
        def IndexFlatIP(dimension):
            return FakeIndex(dimension)

        @staticmethod
        def write_index(index, path):
            with open(path, "w", encoding="utf-8") as file:
                json.dump({"d": index.d, "ntotal": index.ntotal}, file)

        @staticmethod
        def read_index(path):
            with open(path, encoding="utf-8") as file:
                payload = json.load(file)
            index = FakeIndex(payload["d"])
            index.ntotal = payload["ntotal"]
            return index

        @staticmethod
        def omp_set_num_threads(_):
            return None

    monkeypatch.setattr(embed_utils, "_load_faiss", lambda: FakeFaiss)
    knowledge_base = FaissKnowledgeBase(index_dir=tmp_path, embedding_model="test-model")
    knowledge_base.save(
        np.ones((1, 3), dtype="float32"),
        [DocumentChunk(id=0, source="v1", text="上一版")],
        source_signature="v1",
    )
    knowledge_base.save(
        np.ones((1, 3), dtype="float32"),
        [DocumentChunk(id=0, source="v2", text="当前版")],
        source_signature="v2",
    )
    (tmp_path / "metadata.json").write_text("{broken", encoding="utf-8")

    restored = FaissKnowledgeBase(index_dir=tmp_path, embedding_model="test-model")
    assert restored.exists is True
    restored.load()

    assert restored.index_info()["loaded_from_backup"] is True
    assert restored._metadata[0]["text"] == "上一版"
