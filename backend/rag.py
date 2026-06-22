"""Retrieval augmented generation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, Protocol

from backend.llm import LLMClient, build_llm_client
from utils.config import settings
from utils.prompts import SPORTS_HEALTH_SYSTEM_PROMPT


logger = logging.getLogger("uvicorn.error")
INDEX_FILES = ("index.faiss", "metadata.json")


def persisted_index_exists(index_dir: Path) -> bool:
    return all((index_dir / name).exists() for name in INDEX_FILES)


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return top-k document hits."""


class FaissRetriever:
    """Lazy FAISS retriever wrapper."""

    def __init__(
        self,
        index_dir: str | Path,
        embedding_model: str,
        embedding_device: str | None = None,
        embedding_batch_size: int = 32,
        faiss_threads: int = 1,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.embedding_batch_size = embedding_batch_size
        self.faiss_threads = faiss_threads
        self._kb = None

    @property
    def exists(self) -> bool:
        return persisted_index_exists(self.index_dir)

    def _knowledge_base(self):
        if self._kb is None:
            from embeddings.embed_utils import FaissKnowledgeBase

            self._kb = FaissKnowledgeBase(
                index_dir=self.index_dir,
                embedding_model=self.embedding_model,
                device=self.embedding_device,
                batch_size=self.embedding_batch_size,
                faiss_threads=self.faiss_threads,
            )
        return self._kb

    def load(self) -> dict[str, Any]:
        knowledge_base = self._knowledge_base()
        knowledge_base.load()
        return knowledge_base.index_info()

    def close(self) -> None:
        return

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
            lora_adapter_path=settings.active_lora_adapter_path,
            max_new_tokens=settings.local_model_max_new_tokens,
            temperature=settings.local_model_temperature,
            top_p=settings.local_model_top_p,
            enable_thinking=settings.local_model_enable_thinking,
            local_files_only=settings.local_files_only,
        )
        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = FaissRetriever(
                index_dir=settings.faiss_index_dir,
                embedding_model=settings.embedding_model,
                embedding_device=settings.embedding_device,
                embedding_batch_size=settings.embedding_batch_size,
                faiss_threads=settings.faiss_threads,
            )
        self.max_context_chars = max_context_chars
        self._retrieval_enabled = True
        self._retrieval_disabled_reason: str | None = None

    def enable_retrieval(self) -> None:
        self._retrieval_enabled = True
        self._retrieval_disabled_reason = None

    def disable_retrieval(self, reason: str) -> None:
        self._retrieval_enabled = False
        self._retrieval_disabled_reason = reason
        logger.error("Knowledge-base retrieval disabled: %s", reason)

    def retrieve(self, query: str, top_k: int = settings.default_top_k) -> list[dict[str, Any]]:
        if not self._retrieval_enabled:
            return []
        try:
            return self.retriever.search(query, top_k=top_k)
        except Exception:
            logger.exception("Knowledge-base retrieval failed.")
            if settings.retrieval_failure_fallback:
                return []
            raise

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

    def build_chat_prompt(
        self,
        question: str,
        memory_context: str = "",
        enable_thinking: bool | None = None,
    ) -> str:
        memory = (memory_context or "暂无")[-max(1000, self.max_context_chars // 3) :]
        thinking_instruction = (
            "如果启用深度思考，请在 <think></think> 中使用中文进行思考，最终回答也使用中文。\n"
            if enable_thinking
            else ""
        )
        return f"""{SPORTS_HEALTH_SYSTEM_PROMPT}

用户当前是在进行普通对话或寒暄时，请自然回应，不要提及知识库、检索结果或 RAG。
如果用户随后提出专业问题，再结合本地知识库严谨回答。
{thinking_instruction}

【多轮对话记忆】
{memory}

【用户消息】
{question}
"""

    def build_prompt(
        self,
        question: str,
        documents: list[dict[str, Any]],
        memory_context: str = "",
        enable_thinking: bool | None = None,
    ) -> str:
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
        retrieved_context = retrieved_context[:retrieval_budget]
        thinking_instruction = (
            "如果启用深度思考，请在 <think></think> 中使用中文进行思考，最终回答也使用中文。\n"
            if enable_thinking
            else ""
        )

        return f"""{SPORTS_HEALTH_SYSTEM_PROMPT}

请优先依据【本地知识库】回答；如果知识库不足，请明确说明不确定，并给出可验证的建议。
{thinking_instruction}

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
            prompt = self.build_chat_prompt(
                question,
                memory_context=memory_context,
                enable_thinking=enable_thinking,
            )
        else:
            documents = self.retrieve(question, top_k=top_k)
            prompt = self.build_prompt(
                question,
                documents,
                memory_context=memory_context,
                enable_thinking=enable_thinking,
            )
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
            prompt = self.build_chat_prompt(
                question,
                memory_context=memory_context,
                enable_thinking=enable_thinking,
            )
        else:
            documents = self.retrieve(question, top_k=top_k)
            prompt = self.build_prompt(
                question,
                documents,
                memory_context=memory_context,
                enable_thinking=enable_thinking,
            )
        return documents, self.llm_client.stream_generate(prompt, enable_thinking=enable_thinking)
