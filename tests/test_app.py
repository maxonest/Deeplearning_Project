from fastapi.testclient import TestClient

import backend.app as app_module


class FakeLLM:
    def __init__(self):
        self.is_loaded = False

    def load(self) -> None:
        self.is_loaded = True

    def generate(self, prompt: str, enable_thinking: bool | None = None) -> str:
        return f"模型回答:{prompt}"


class FakePipeline:
    def __init__(self):
        self.llm_client = FakeLLM()
        self.retriever = FakeRetriever()

    def answer(
        self,
        question: str,
        memory_context: str = "",
        top_k: int = 5,
        enable_thinking: bool | None = None,
    ):
        return {
            "answer": f"回答:{question}",
            "documents": [{"id": 0, "source": "fake", "text": memory_context, "score": 1.0}],
            "prompt": "prompt",
        }


class FakeRetriever:
    exists = False

    def __init__(self):
        self.rebuild_sources = []

    def rebuild(self, input_paths, chunk_size, overlap):
        self.rebuild_sources = input_paths
        self.exists = True
        return 12

    def search(self, query, top_k):
        return [{"source": "fake-sft.json#0", "text": "测试", "score": 1.0}]


def test_health_endpoint():
    client = TestClient(app_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()
    assert "knowledge_base_ready" in response.json()
    assert "knowledge_base_chunks" in response.json()


def test_model_endpoint_calls_llm_directly(monkeypatch):
    monkeypatch.setattr(app_module.settings, "use_local_model", True)
    pipeline = FakePipeline()
    pipeline.llm_client.load()
    monkeypatch.setattr(app_module, "rag_pipeline", pipeline)
    client = TestClient(app_module.app)

    response = client.post("/api/model/test", json={"prompt": "测试LoRA"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "模型回答:测试LoRA"
    assert payload["model_loaded"] is True


def test_preload_local_model_loads_configured_client(monkeypatch):
    pipeline = FakePipeline()
    monkeypatch.setattr(app_module.settings, "use_local_model", True)
    monkeypatch.setattr(app_module, "rag_pipeline", pipeline)

    app_module.preload_local_model()

    assert pipeline.llm_client.is_loaded is True


def test_rebuild_knowledge_base_includes_finetune_dataset(monkeypatch, tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    finetune_path = tmp_path / "sft_dataset_clean.json"
    finetune_path.write_text("[]", encoding="utf-8")
    pipeline = FakePipeline()

    monkeypatch.setattr(app_module.settings, "rebuild_knowledge_base_on_startup", True)
    monkeypatch.setattr(app_module.settings, "processed_data_dir", processed_dir)
    monkeypatch.setattr(app_module.settings, "finetune_dataset_path", finetune_path)
    monkeypatch.setattr(app_module, "rag_pipeline", pipeline)

    app_module.rebuild_knowledge_base()

    assert pipeline.retriever.rebuild_sources == [processed_dir, finetune_path]
    assert app_module.startup_state["knowledge_base_ready"] is True
    assert app_module.startup_state["knowledge_base_chunks"] == 12


def test_chat_endpoint_uses_pipeline(monkeypatch):
    monkeypatch.setattr(app_module, "rag_pipeline", FakePipeline())
    client = TestClient(app_module.app)

    response = client.post("/api/chat", json={"question": "测试问题", "top_k": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "回答:测试问题"
    assert payload["session_id"]
    assert payload["documents"][0]["source"] == "fake"
