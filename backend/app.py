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
startup_state = {
    "knowledge_base_ready": False,
    "knowledge_base_chunks": 0,
    "knowledge_base_error": None,
}


def rebuild_knowledge_base() -> None:
    startup_state["knowledge_base_ready"] = False
    startup_state["knowledge_base_chunks"] = 0
    startup_state["knowledge_base_error"] = None
    sources = [settings.processed_data_dir]
    if settings.finetune_dataset_path.is_file():
        sources.append(settings.finetune_dataset_path)
    else:
        logger.warning(
            "Fine-tuning dataset was not found and will not be indexed: %s",
            settings.finetune_dataset_path,
        )

    retriever = rag_pipeline.retriever
    try:
        load = getattr(retriever, "load", None)
        rebuild = getattr(retriever, "rebuild", None)
        if load is None:
            raise RuntimeError("The configured retriever does not support index validation.")
        if rebuild is None:
            raise RuntimeError("The configured retriever does not support rebuilding.")

        index_info = None
        invalid_index_error = None
        try:
            index_info = load()
        except Exception as exc:
            invalid_index_error = exc
            logger.warning(
                "Existing knowledge-base index is missing or incompatible and must be rebuilt: %s",
                exc,
            )

        should_rebuild = invalid_index_error is not None
        rebuild_reason = "missing, legacy, or incompatible index"
        if not should_rebuild and settings.rebuild_knowledge_base_on_startup:
            needs_rebuild = getattr(retriever, "needs_rebuild", None)
            if needs_rebuild is None:
                raise RuntimeError("The configured retriever cannot check index freshness.")
            should_rebuild = needs_rebuild(
                input_paths=sources,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            rebuild_reason = "source content or indexing configuration changed"

        if should_rebuild:
            logger.info(
                "Knowledge-base stage 1/3: %s; rebuilding "
                "from sources=%s, embedding_model=%s, device=%s",
                rebuild_reason,
                [str(source) for source in sources],
                settings.embedding_model,
                settings.embedding_device,
            )
            chunk_count = rebuild(
                input_paths=sources,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            logger.info(
                "Knowledge-base stage 2/3: saved %s chunks to %s",
                chunk_count,
                settings.faiss_index_dir,
            )
        else:
            logger.info("Knowledge-base stage 1/3: existing index is current; skipping rebuild.")
            assert index_info is not None
            chunk_count = int(index_info.get("count", 0))
            logger.info(
                "Knowledge-base stage 2/3: validated existing index with %s chunks.",
                chunk_count,
            )

        startup_state["knowledge_base_chunks"] = chunk_count
        hits = retriever.search(settings.knowledge_base_self_test_query, top_k=1)
        if not hits:
            raise RuntimeError(
                "Knowledge-base self-test returned no results after loading the index."
        )
        startup_state["knowledge_base_ready"] = True
        rag_pipeline.enable_retrieval()
        logger.info(
            "Knowledge-base stage 3/3: self-test succeeded. query=%r, source=%s",
            settings.knowledge_base_self_test_query,
            hits[0].get("source", "unknown"),
        )
    except Exception as exc:
        startup_state["knowledge_base_error"] = f"{type(exc).__name__}: {exc}"
        rag_pipeline.disable_retrieval(startup_state["knowledge_base_error"])
        logger.exception("Knowledge-base startup preparation failed.")
        if not settings.retrieval_failure_fallback:
            raise
        logger.warning(
            "Continuing without retrieval because RETRIEVAL_FAILURE_FALLBACK=true."
        )


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
    rebuild_knowledge_base()
    preload_local_model()
    try:
        yield
    finally:
        close_retriever = getattr(rag_pipeline.retriever, "close", None)
        if close_retriever is not None:
            close_retriever()


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
        status="ok" if startup_state["knowledge_base_ready"] else "degraded",
        use_local_model=settings.use_local_model,
        model_loaded=bool(getattr(rag_pipeline.llm_client, "is_loaded", False)),
        knowledge_base_ready=bool(startup_state["knowledge_base_ready"]),
        knowledge_base_chunks=int(startup_state["knowledge_base_chunks"]),
        knowledge_base_error=startup_state["knowledge_base_error"],
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

    memory_context = memory.build_context(session_id)
    memory.add_message(session_id, "user", request.question)
    result = rag_pipeline.answer(
        question=request.question,
        memory_context=memory_context,
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

        memory_context = memory.build_context(session_id)
        memory.add_message(session_id, "user", request.question)
        answer_parts: list[str] = []

        try:
            documents, chunks = rag_pipeline.stream_answer(
                question=request.question,
                memory_context=memory_context,
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
            logger.exception("Streaming chat request failed.")
            yield sse_event("error", {"message": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post(f"{settings.api_prefix}/sessions/{{session_id}}/clear")
def clear_session(session_id: str) -> dict[str, str]:
    memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}
