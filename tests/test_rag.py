from backend.rag import RAGPipeline


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
