import backend.rag as rag_module
from backend.rag import RAGPipeline
from utils.config import settings


class FakeRetriever:
    def search(self, query: str, top_k: int):
        return [{"id": 1, "source": "doc.txt", "text": f"知识:{query}", "score": 0.9}][:top_k]


class FakeLLM:
    def __init__(self):
        self.last_prompt = ""

    def generate(self, prompt: str, enable_thinking: bool | None = None) -> str:
        self.last_prompt = prompt
        return "测试回答"


def test_rag_pipeline_builds_prompt_and_calls_llm():
    llm = FakeLLM()
    pipeline = RAGPipeline(llm_client=llm, retriever=FakeRetriever())

    result = pipeline.answer("什么是RAG？", memory_context="user: 你好", top_k=1)

    assert result["answer"] == "测试回答"
    assert result["documents"][0]["source"] == "doc.txt"
    assert "什么是RAG？" in llm.last_prompt
    assert "user: 你好" in llm.last_prompt
    assert "知识:什么是RAG？" in llm.last_prompt
    assert "运动健康垂直领域专家" in llm.last_prompt


def test_rag_context_truncation_keeps_highest_ranked_document():
    pipeline = RAGPipeline(llm_client=FakeLLM(), retriever=FakeRetriever(), max_context_chars=1200)
    documents = [
        {"source": "rank-1", "text": "最相关内容" * 300, "score": 0.99},
        {"source": "rank-2", "text": "次相关内容" * 300, "score": 0.80},
    ]

    prompt = pipeline.build_prompt("问题", documents)

    assert "rank-1" in prompt


def test_retrieval_failure_falls_back_to_empty_documents(monkeypatch):
    retriever = FakeRetriever()
    monkeypatch.setattr(
        retriever,
        "search",
        lambda query, top_k: (_ for _ in ()).throw(RuntimeError("native crash")),
    )
    monkeypatch.setattr(settings, "retrieval_failure_fallback", True)
    pipeline = RAGPipeline(llm_client=FakeLLM(), retriever=retriever)

    assert pipeline.retrieve("专业问题", top_k=1) == []


def test_disabled_retrieval_does_not_call_worker():
    retriever = FakeRetriever()
    retriever.search = lambda query, top_k: (_ for _ in ()).throw(
        AssertionError("worker should not be called")
    )
    pipeline = RAGPipeline(llm_client=FakeLLM(), retriever=retriever)

    pipeline.disable_retrieval("startup rebuild failed")

    assert pipeline.retrieve("专业问题", top_k=1) == []


def test_rag_pipeline_respects_disabled_lora_switch(monkeypatch, tmp_path):
    captured = {}

    def fake_build_llm_client(**kwargs):
        captured.update(kwargs)
        return FakeLLM()

    monkeypatch.setattr(rag_module, "build_llm_client", fake_build_llm_client)
    monkeypatch.setattr(settings, "use_lora_adapter", False)
    monkeypatch.setattr(settings, "local_lora_adapter_path", tmp_path / "lora_adapter")

    RAGPipeline(retriever=FakeRetriever())

    assert captured["lora_adapter_path"] is None


def test_disabled_rag_skips_retrieval_and_uses_direct_prompt():
    retriever = FakeRetriever()
    retriever.search = lambda query, top_k: (_ for _ in ()).throw(
        AssertionError("RAG-disabled requests must not retrieve")
    )
    llm = FakeLLM()
    pipeline = RAGPipeline(llm_client=llm, retriever=retriever, use_rag=False)

    result = pipeline.answer("如何安排每周运动计划？", top_k=5)

    assert result["documents"] == []
    assert "如何安排每周运动计划？" in llm.last_prompt
    assert "本地知识库" not in llm.last_prompt
    assert "避免机械复述固定模板" in llm.last_prompt
