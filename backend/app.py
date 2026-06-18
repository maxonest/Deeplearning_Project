"""FastAPI application for local domain QA."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.memory import ConversationMemory
from backend.rag import RAGPipeline
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    ConfigResponse,
    HealthResponse,
    ModelTestRequest,
    ModelTestResponse,
    SessionResponse,
)
from utils.config import settings


logger = logging.getLogger("uvicorn.error")

memory = ConversationMemory(
    max_recent_turns=settings.max_recent_turns,
    max_context_chars=settings.max_context_chars,
)
rag_pipeline = RAGPipeline()


def preload_local_model() -> None:
    if not settings.use_local_model:
        return

    adapter = (
        str(settings.local_lora_adapter_path)
        if settings.local_lora_adapter_path
        else "disabled"
    )
    logger.info(
        "Loading local model before accepting requests. base_model=%s, lora_adapter=%s",
        settings.local_model_path,
        adapter,
    )
    rag_pipeline.llm_client.load()
    logger.info("Local model startup loading completed.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    preload_local_model()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        use_local_model=settings.use_local_model,
        model_loaded=bool(getattr(rag_pipeline.llm_client, "is_loaded", False)),
        local_model_path=str(settings.local_model_path),
        local_lora_adapter_path=(
            str(settings.local_lora_adapter_path) if settings.local_lora_adapter_path else None
        ),
    )


@app.post(f"{settings.api_prefix}/model/test", response_model=ModelTestResponse)
def test_model(request: ModelTestRequest) -> ModelTestResponse:
    if not settings.use_local_model:
        raise HTTPException(
            status_code=503,
            detail="Local model is disabled. Set USE_LOCAL_MODEL=true in .env and restart the backend.",
        )

    answer = rag_pipeline.llm_client.generate(
        request.prompt,
        enable_thinking=request.enable_thinking,
    )
    return ModelTestResponse(
        answer=answer,
        model_loaded=bool(getattr(rag_pipeline.llm_client, "is_loaded", False)),
        local_model_path=str(settings.local_model_path),
        local_lora_adapter_path=(
            str(settings.local_lora_adapter_path) if settings.local_lora_adapter_path else None
        ),
    )


@app.get(f"{settings.api_prefix}/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    return ConfigResponse(
        app_name=settings.app_name,
        default_top_k=settings.default_top_k,
        use_local_model=settings.use_local_model,
        local_model_path=str(settings.local_model_path),
        local_lora_adapter_path=(
            str(settings.local_lora_adapter_path) if settings.local_lora_adapter_path else None
        ),
        embedding_model=settings.embedding_model,
    )


@app.post(f"{settings.api_prefix}/sessions", response_model=SessionResponse)
def create_session() -> SessionResponse:
    return SessionResponse(session_id=memory.create_session())


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or memory.create_session()
    if not memory.has_session(session_id):
        memory.clear(session_id)

    memory.add_message(session_id, "user", request.question)
    result = rag_pipeline.answer(
        question=request.question,
        memory_context=memory.build_context(session_id),
        top_k=request.top_k,
        enable_thinking=request.enable_thinking,
    )
    answer = result["answer"]
    memory.add_message(session_id, "assistant", answer)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        documents=result["documents"],
    )


def sse_event(event: str, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.post(f"{settings.api_prefix}/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def event_generator():
        session_id = request.session_id or memory.create_session()
        if not memory.has_session(session_id):
            memory.clear(session_id)

        memory.add_message(session_id, "user", request.question)
        answer_parts: list[str] = []

        try:
            documents, chunks = rag_pipeline.stream_answer(
                question=request.question,
                memory_context=memory.build_context(session_id),
                top_k=request.top_k,
                enable_thinking=request.enable_thinking,
            )
            yield sse_event("meta", {"session_id": session_id, "documents": documents})
            for chunk in chunks:
                answer_parts.append(chunk)
                yield sse_event("delta", {"text": chunk})
            answer = "".join(answer_parts).strip()
            memory.add_message(session_id, "assistant", answer)
            yield sse_event("done", {"answer": answer})
        except Exception as exc:
            yield sse_event("error", {"message": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post(f"{settings.api_prefix}/sessions/{{session_id}}/clear")
def clear_session(session_id: str) -> dict[str, str]:
    memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}
