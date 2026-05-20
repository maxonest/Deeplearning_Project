"""Local RAG retrieval and prompt assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from embeddings.embed_utils import search
from utils.config import settings


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate an answer from a prompt."""


class LocalLLMClient:
    """Placeholder local LLM client.

    Replace this class with a vLLM, Ollama, llama.cpp, or Transformers backend.
    """

    def generate(self, prompt: str) -> str:
        return (
            "这是一个本地模型占位回答。请将 backend/rag.py 中的 LocalLLMClient "
            "替换为实际的大模型推理服务。\n\n"
            f"已接收提示词长度: {len(prompt)} 字符"
        )


class RAGPipeline:
    def __init__(
        self,
        index_dir: str | Path = settings.faiss_index_dir,
        embedding_model: str = settings.embedding_model,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.llm_client = llm_client or LocalLLMClient()

    def retrieve(self, query: str, top_k: int = settings.default_top_k) -> list[dict]:
        if not (self.index_dir / "index.faiss").exists():
            return []
        return search(
            query=query,
            index_dir=self.index_dir,
            model_name=self.embedding_model,
            top_k=top_k,
        )

    def build_prompt(self, question: str, documents: list[dict], memory_context: str = "") -> str:
        context_blocks = []
        for idx, doc in enumerate(documents, start=1):
            source = doc.get("source", "unknown")
            text = doc.get("text", "")
            context_blocks.append(f"[文档 {idx}] 来源: {source}\n{text}")

        retrieved_context = "\n\n".join(context_blocks) or "未检索到本地知识库内容。"

        return f"""你是一个专业、严谨的本地领域知识问答助手。
请优先依据【本地知识库】回答；如果知识库不足，请明确说明不确定，并给出可验证的建议。

【多轮对话记忆】
{memory_context or "暂无"}

【本地知识库】
{retrieved_context}

【用户问题】
{question}

【回答要求】
1. 回答要准确、结构清晰。
2. 涉及事实时尽量引用文档来源。
3. 不要编造知识库中不存在的关键事实。
"""

    def answer(self, question: str, memory_context: str = "", top_k: int = settings.default_top_k) -> dict:
        documents = self.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, documents, memory_context=memory_context)
        answer = self.llm_client.generate(prompt)
        return {
            "answer": answer,
            "documents": documents,
            "prompt": prompt,
        }
