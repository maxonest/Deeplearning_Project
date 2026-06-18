import json

from embeddings.embed_utils import build_chunks_from_sources


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
