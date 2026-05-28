import json

import pytest

from utils.data_loader import format_sft_prompt, load_json_records, normalize_record, split_records


def test_load_json_records_accepts_array_and_normalizes(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps([{"question": "什么是RAG？", "answer": "检索增强生成。"}], ensure_ascii=False),
        encoding="utf-8",
    )

    records = load_json_records(path)

    assert records == [
        {
            "instruction": "根据专业知识回答用户问题",
            "input": "什么是RAG？",
            "output": "检索增强生成。",
            "metadata": {},
        }
    ]


def test_normalize_record_requires_input_and_output():
    with pytest.raises(ValueError):
        normalize_record({"question": "缺少答案"})


def test_split_records_is_deterministic():
    records = [{"input": str(index), "output": str(index), "instruction": "i", "metadata": {}} for index in range(10)]

    first = split_records(records, seed=123)
    second = split_records(records, seed=123)

    assert first == second
    assert len(first["train"]) == 8
    assert len(first["validation"]) == 1
    assert len(first["test"]) == 1


def test_format_sft_prompt_contains_chatml_parts():
    prompt = format_sft_prompt({"instruction": "回答", "input": "问题", "output": "答案"})

    assert "<|im_start|>system" in prompt
    assert "回答" in prompt
    assert "问题" in prompt
    assert "答案" in prompt
