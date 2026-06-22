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
        self.last_memory_context = None
        self.retrieval_enabled = True

    def answer(
        self,
        question: str,
        memory_context: str = "",
        top_k: int = 5,
        enable_thinking: bool | None = None,
    ):
        self.last_memory_context = memory_context
        return {
            "answer": f"回答:{question}",
            "documents": [{"id": 0, "source": "fake", "text": memory_context, "score": 1.0}],
            "prompt": "prompt",
        }

    def enable_retrieval(self):
        self.retrieval_enabled = True

    def disable_retrieval(self, reason):
        self.retrieval_enabled = False


class FakeRetriever:
    exists = False

    def load(self):
        return {"count": 12, "dimension": 384}

    def close(self):
        return

    def search(self, query, top_k):
        return [{"source": "fake-sft.json#0", "text": "测试", "score": 1.0}]


def set_runtime_ready():
    app_module.update_startup_state(
        startup_phase="ready",
        startup_ready=True,
        startup_message="系统已就绪",
        startup_error=None,
    )


def test_health_endpoint():
    client = TestClient(app_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded", "initializing", "failed"}
    assert "startup_phase" in response.json()
    assert "startup_ready" in response.json()
    assert "startup_message" in response.json()
    assert "startup_error" in response.json()
    assert "model_loaded" in response.json()
    assert "knowledge_base_ready" in response.json()
    assert "knowledge_base_chunks" in response.json()
    assert "knowledge_base_error" in response.json()


def test_model_endpoint_calls_llm_directly(monkeypatch):
    set_runtime_ready()
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


def test_load_knowledge_base_validates_persisted_index(monkeypatch):
    pipeline = FakePipeline()
    monkeypatch.setattr(app_module, "rag_pipeline", pipeline)

    app_module.load_knowledge_base()

    assert app_module.startup_state["knowledge_base_ready"] is True
    assert app_module.startup_state["knowledge_base_chunks"] == 12
    assert pipeline.retrieval_enabled is True


def test_invalid_index_fails_without_rebuilding(monkeypatch):
    pipeline = FakePipeline()
    monkeypatch.setattr(
        pipeline.retriever,
        "load",
        lambda: (_ for _ in ()).throw(RuntimeError("invalid local index")),
    )
    monkeypatch.setattr(app_module, "rag_pipeline", pipeline)
    monkeypatch.setattr(app_module.settings, "retrieval_failure_fallback", False)

    try:
        app_module.load_knowledge_base()
    except RuntimeError as exc:
        assert "python embeddings/embed_utils.py build" in str(exc)
    else:
        raise AssertionError("Invalid persisted index must fail startup validation.")

    assert pipeline.retrieval_enabled is False
    assert "invalid local index" in app_module.startup_state["knowledge_base_error"]


def test_invalid_index_can_fall_back_to_model_only(monkeypatch):
    pipeline = FakePipeline()
    monkeypatch.setattr(
        pipeline.retriever,
        "load",
        lambda: (_ for _ in ()).throw(RuntimeError("invalid local index")),
    )
    monkeypatch.setattr(app_module, "rag_pipeline", pipeline)
    monkeypatch.setattr(app_module.settings, "retrieval_failure_fallback", True)

    app_module.load_knowledge_base()

    assert app_module.startup_state["knowledge_base_ready"] is False
    assert pipeline.retrieval_enabled is False


def test_chat_endpoint_uses_pipeline(monkeypatch):
    set_runtime_ready()
    pipeline = FakePipeline()
    monkeypatch.setattr(app_module, "rag_pipeline", pipeline)
    client = TestClient(app_module.app)

    response = client.post("/api/chat", json={"question": "测试问题", "top_k": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "回答:测试问题"
    assert payload["session_id"]
    assert payload["documents"][0]["source"] == "fake"
    assert "测试问题" not in (pipeline.last_memory_context or "")


def test_chat_endpoint_rejects_requests_until_runtime_is_ready():
    app_module.update_startup_state(
        startup_phase="model",
        startup_ready=False,
        startup_message="模型加载中",
        startup_error=None,
    )
    client = TestClient(app_module.app)

    response = client.post("/api/chat", json={"question": "你好"})

    assert response.status_code == 503
    assert response.json()["detail"] == "模型加载中"


def test_initialize_runtime_exposes_model_phase_and_ready_state(monkeypatch):
    phases = []
    monkeypatch.setattr(
        app_module,
        "load_knowledge_base",
        lambda: phases.append(app_module.get_startup_state()["startup_phase"]),
    )
    monkeypatch.setattr(
        app_module,
        "preload_local_model",
        lambda: phases.append(app_module.get_startup_state()["startup_phase"]),
    )
    monkeypatch.setattr(app_module.settings, "use_local_model", True)
    app_module.update_startup_state(knowledge_base_ready=True)

    app_module.initialize_runtime()

    state = app_module.get_startup_state()
    assert phases == ["knowledge_base", "model"]
    assert state["startup_phase"] == "ready"
    assert state["startup_ready"] is True
