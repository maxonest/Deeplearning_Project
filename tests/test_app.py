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

    def __init__(self):
        self.rebuild_sources = []

    def rebuild(self, input_paths, chunk_size, overlap):
        self.rebuild_sources = input_paths
        self.exists = True
        return 12

    def needs_rebuild(self, input_paths, chunk_size, overlap):
        return True

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


def test_invalid_legacy_index_is_rebuilt_even_when_auto_update_is_disabled(monkeypatch, tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pipeline = FakePipeline()
    rebuild_calls = []

    monkeypatch.setattr(
        pipeline.retriever,
        "load",
        lambda: (_ for _ in ()).throw(
            RuntimeError("Unsupported FAISS metadata format: None")
        ),
    )
    monkeypatch.setattr(
        pipeline.retriever,
        "rebuild",
        lambda input_paths, chunk_size, overlap: rebuild_calls.append(input_paths) or 8,
    )
    monkeypatch.setattr(app_module.settings, "rebuild_knowledge_base_on_startup", False)
    monkeypatch.setattr(app_module.settings, "processed_data_dir", processed_dir)
    monkeypatch.setattr(app_module.settings, "finetune_dataset_path", tmp_path / "missing.json")
    monkeypatch.setattr(app_module, "rag_pipeline", pipeline)

    app_module.rebuild_knowledge_base()

    assert rebuild_calls == [[processed_dir]]
    assert app_module.startup_state["knowledge_base_ready"] is True
    assert pipeline.retrieval_enabled is True


def test_failed_rebuild_disables_repeated_retrieval(monkeypatch, tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pipeline = FakePipeline()

    monkeypatch.setattr(
        pipeline.retriever,
        "load",
        lambda: (_ for _ in ()).throw(RuntimeError("legacy index")),
    )
    monkeypatch.setattr(
        pipeline.retriever,
        "rebuild",
        lambda input_paths, chunk_size, overlap: (_ for _ in ()).throw(
            RuntimeError("rebuild failed")
        ),
    )
    monkeypatch.setattr(app_module.settings, "retrieval_failure_fallback", True)
    monkeypatch.setattr(app_module.settings, "processed_data_dir", processed_dir)
    monkeypatch.setattr(app_module.settings, "finetune_dataset_path", tmp_path / "missing.json")
    monkeypatch.setattr(app_module, "rag_pipeline", pipeline)

    app_module.rebuild_knowledge_base()

    assert pipeline.retrieval_enabled is False
    assert "rebuild failed" in app_module.startup_state["knowledge_base_error"]


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
        "rebuild_knowledge_base",
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
