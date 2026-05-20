"""FastAPI backend entry point for local domain QA."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.memory import ConversationMemory
from backend.rag import RAGPipeline
from utils.config import settings


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory = ConversationMemory(
    max_recent_turns=settings.max_recent_turns,
    max_context_chars=settings.max_context_chars,
)
rag_pipeline = RAGPipeline()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None
    top_k: int = Field(default=settings.default_top_k, ge=1, le=20)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    documents: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or memory.create_session()
    memory.add_message(session_id, "user", request.question)

    memory_context = memory.build_context(session_id)
    result = rag_pipeline.answer(
        question=request.question,
        memory_context=memory_context,
        top_k=request.top_k,
    )

    answer = result["answer"]
    memory.add_message(session_id, "assistant", answer)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        documents=result["documents"],
    )


@app.post(f"{settings.api_prefix}/sessions/{{session_id}}/clear")
def clear_session(session_id: str) -> dict[str, str]:
    memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}
