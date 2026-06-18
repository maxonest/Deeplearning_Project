import sys
from types import SimpleNamespace

import pytest

from backend.llm import TransformersLLMClient


class FakeModel:
    def __init__(self):
        self.eval_called = False
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def test_load_mounts_lora_adapter(monkeypatch, tmp_path):
    model_path = tmp_path / "base_model"
    adapter_path = tmp_path / "lora_adapter"
    model_path.mkdir()
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_path / "adapter_model.safetensors").write_bytes(b"weights")

    base_model = FakeModel()
    peft_model = FakeModel()
    calls = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["tokenizer"] = (path, kwargs)
            return object()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["base_model"] = (path, kwargs)
            return base_model

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, path, **kwargs):
            calls["adapter"] = (model, path, kwargs)
            return peft_model

    fake_torch = SimpleNamespace(
        __version__="test",
        bfloat16="bfloat16",
        float32="float32",
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
    )
    fake_transformers = SimpleNamespace(
        __version__="test",
        AutoModelForCausalLM=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
    )
    fake_peft = SimpleNamespace(__version__="test", PeftModel=FakePeftModel)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    client = TransformersLLMClient(
        model_path=model_path,
        lora_adapter_path=adapter_path,
    )
    client.load()

    assert calls["adapter"][0] is base_model
    assert calls["adapter"][1] == str(adapter_path)
    assert calls["adapter"][2]["local_files_only"] is True
    assert client._model is peft_model
    assert peft_model.device == "cpu"
    assert peft_model.eval_called is True


def test_load_rejects_incomplete_lora_adapter(tmp_path):
    model_path = tmp_path / "base_model"
    adapter_path = tmp_path / "lora_adapter"
    model_path.mkdir()
    adapter_path.mkdir()

    client = TransformersLLMClient(
        model_path=model_path,
        lora_adapter_path=adapter_path,
    )

    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        client.load()
