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


def test_health_endpoint():
    client = TestClient(app_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()


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


def test_chat_endpoint_uses_pipeline(monkeypatch):
    monkeypatch.setattr(app_module, "rag_pipeline", FakePipeline())
    client = TestClient(app_module.app)

    response = client.post("/api/chat", json={"question": "测试问题", "top_k": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "回答:测试问题"
    assert payload["session_id"]
    assert payload["documents"][0]["source"] == "fake"
