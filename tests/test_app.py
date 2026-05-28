from fastapi.testclient import TestClient

import backend.app as app_module


class FakePipeline:
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


def test_chat_endpoint_uses_pipeline(monkeypatch):
    monkeypatch.setattr(app_module, "rag_pipeline", FakePipeline())
    client = TestClient(app_module.app)

    response = client.post("/api/chat", json={"question": "测试问题", "top_k": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "回答:测试问题"
    assert payload["session_id"]
    assert payload["documents"][0]["source"] == "fake"
