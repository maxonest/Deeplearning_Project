"""Retrieval augmented generation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Protocol

from backend.llm import LLMClient, build_llm_client
from utils.config import settings


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return top-k document hits."""


class FaissRetriever:
    """Lazy FAISS retriever wrapper."""

    def __init__(self, index_dir: str | Path, embedding_model: str) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self._kb = None

    @property
    def exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "metadata.json").exists()

    def _knowledge_base(self):
        if self._kb is None:
            from embeddings.embed_utils import FaissKnowledgeBase

            self._kb = FaissKnowledgeBase(index_dir=self.index_dir, embedding_model=self.embedding_model)
        return self._kb

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self.exists:
            return []
        return self._knowledge_base().search(query, top_k=top_k)


class RAGPipeline:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        retriever: Retriever | None = None,
        max_context_chars: int = settings.max_context_chars,
    ) -> None:
        self.llm_client = llm_client or build_llm_client(
            use_local_model=settings.use_local_model,
            model_path=settings.local_model_path,
            max_new_tokens=settings.local_model_max_new_tokens,
            temperature=settings.local_model_temperature,
            top_p=settings.local_model_top_p,
            enable_thinking=settings.local_model_enable_thinking,
            local_files_only=settings.local_files_only,
        )
        self.retriever = retriever or FaissRetriever(
            index_dir=settings.faiss_index_dir,
            embedding_model=settings.embedding_model,
        )
        self.max_context_chars = max_context_chars

    def retrieve(self, query: str, top_k: int = settings.default_top_k) -> list[dict[str, Any]]:
        return self.retriever.search(query, top_k=top_k)

    @staticmethod
    def is_small_talk(question: str) -> bool:
        normalized = question.strip().lower().replace("！", "").replace("!", "").replace("。", "")
        small_talk = {
            "你好",
            "您好",
            "hello",
            "hi",
            "嗨",
            "在吗",
            "你是谁",
            "介绍一下你自己",
            "谢谢",
            "thank you",
            "thanks",
        }
        return normalized in small_talk or (len(normalized) <= 8 and any(word in normalized for word in small_talk))

    def build_chat_prompt(self, question: str, memory_context: str = "") -> str:
        memory = (memory_context or "暂无")[-max(1000, self.max_context_chars // 3) :]
        return f"""你是一个专业、友好、简洁的本地领域知识问答助手。
用户当前是在进行普通对话或寒暄时，请自然回应，不要提及知识库、检索结果或 RAG。
如果用户随后提出专业问题，再结合本地知识库严谨回答。

【多轮对话记忆】
{memory}

【用户消息】
{question}
"""

    def build_prompt(self, question: str, documents: list[dict[str, Any]], memory_context: str = "") -> str:
        context_blocks = []
        for index, document in enumerate(documents, start=1):
            source = document.get("source", "unknown")
            text = document.get("text", "")
            score = document.get("score")
            score_text = f" | score={score:.4f}" if isinstance(score, float) else ""
            context_blocks.append(f"[文档 {index}] 来源: {source}{score_text}\n{text}")

        retrieved_context = "\n\n".join(context_blocks) or "暂无可引用的补充资料。"
        memory_budget = max(1000, self.max_context_chars // 3)
        retrieval_budget = max(1000, self.max_context_chars - memory_budget)
        memory = (memory_context or "暂无")[-memory_budget:]
        retrieved_context = retrieved_context[-retrieval_budget:]

        return f"""你是一个专业、严谨的本地领域知识问答助手。
请优先依据【本地知识库】回答；如果知识库不足，请明确说明不确定，并给出可验证的建议。

【多轮对话记忆】
{memory}

【本地知识库】
{retrieved_context}

【用户问题】
{question}

【回答要求】
1. 回答要准确、结构清晰。
2. 涉及事实时尽量引用文档来源。
3. 不要编造知识库中不存在的关键事实。
"""

    def answer(
        self,
        question: str,
        memory_context: str = "",
        top_k: int = settings.default_top_k,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        if self.is_small_talk(question):
            documents: list[dict[str, Any]] = []
            prompt = self.build_chat_prompt(question, memory_context=memory_context)
        else:
            documents = self.retrieve(question, top_k=top_k)
            prompt = self.build_prompt(question, documents, memory_context=memory_context)
        answer = self.llm_client.generate(prompt, enable_thinking=enable_thinking)
        return {
            "answer": answer,
            "documents": documents,
            "prompt": prompt,
        }

    def stream_answer(
        self,
        question: str,
        memory_context: str = "",
        top_k: int = settings.default_top_k,
        enable_thinking: bool | None = None,
    ) -> tuple[list[dict[str, Any]], Iterator[str]]:
        if self.is_small_talk(question):
            documents: list[dict[str, Any]] = []
            prompt = self.build_chat_prompt(question, memory_context=memory_context)
        else:
            documents = self.retrieve(question, top_k=top_k)
            prompt = self.build_prompt(question, documents, memory_context=memory_context)
        return documents, self.llm_client.stream_generate(prompt, enable_thinking=enable_thinking)
