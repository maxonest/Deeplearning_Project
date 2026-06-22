"""FastAPI application for local domain QA."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from threading import RLock, Thread

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
    "startup_phase": "starting",
    "startup_ready": False,
    "startup_message": "后端正在启动",
    "startup_error": None,
    "knowledge_base_ready": False,
    "knowledge_base_chunks": 0,
    "knowledge_base_error": None,
}
startup_state_lock = RLock()


def update_startup_state(**values) -> None:
    with startup_state_lock:
        startup_state.update(values)


def get_startup_state() -> dict:
    with startup_state_lock:
        return dict(startup_state)


def load_knowledge_base() -> None:
    update_startup_state(
        knowledge_base_ready=False,
        knowledge_base_chunks=0,
        knowledge_base_error=None,
    )
    retriever = rag_pipeline.retriever
    try:
        load = getattr(retriever, "load", None)
        if load is None:
            raise RuntimeError("The configured retriever does not support index validation.")
        logger.info("Knowledge-base stage 1/2: validating persisted local index.")
        index_info = load()
        chunk_count = int(index_info.get("count", 0))
        logger.info("Knowledge-base stage 1/2: loaded %s chunks.", chunk_count)

        update_startup_state(knowledge_base_chunks=chunk_count)
        hits = retriever.search(settings.knowledge_base_self_test_query, top_k=1)
        if not hits:
            raise RuntimeError(
                "Knowledge-base self-test returned no results after loading the index."
            )
        update_startup_state(knowledge_base_ready=True)
        rag_pipeline.enable_retrieval()
        logger.info(
            "Knowledge-base stage 2/2: self-test succeeded. query=%r, source=%s",
            settings.knowledge_base_self_test_query,
            hits[0].get("source", "unknown"),
        )
    except Exception as exc:
        knowledge_base_error = f"{type(exc).__name__}: {exc}"
        update_startup_state(knowledge_base_error=knowledge_base_error)
        rag_pipeline.disable_retrieval(knowledge_base_error)
        logger.exception("Knowledge-base startup preparation failed.")
        if settings.retrieval_failure_fallback:
            logger.warning(
                "Backend will continue without RAG because RETRIEVAL_FAILURE_FALLBACK=true."
            )
            return
        raise RuntimeError(
            "Persisted knowledge base is unavailable. Build it before starting the backend: "
            "python embeddings/embed_utils.py build"
        ) from exc


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


def initialize_runtime() -> None:
    try:
        update_startup_state(
            startup_phase="knowledge_base",
            startup_ready=False,
            startup_message="知识库校验中",
            startup_error=None,
        )
        load_knowledge_base()

        if settings.use_local_model:
            update_startup_state(
                startup_phase="model",
                startup_message="模型加载中",
            )
            preload_local_model()

        state = get_startup_state()
        ready_message = (
            "系统已就绪"
            if state["knowledge_base_ready"]
            else "模型已就绪，知识库暂不可用"
        )
        update_startup_state(
            startup_phase="ready",
            startup_ready=True,
            startup_message=ready_message,
        )
    except Exception as exc:
        logger.exception("Backend runtime initialization failed.")
        update_startup_state(
            startup_phase="failed",
            startup_ready=False,
            startup_message="后端初始化失败",
            startup_error=f"{type(exc).__name__}: {exc}",
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    update_startup_state(
        startup_phase="starting",
        startup_ready=False,
        startup_message="后端正在启动",
        startup_error=None,
    )
    initialization_thread = Thread(
        target=initialize_runtime,
        name="runtime-initializer",
        daemon=True,
    )
    initialization_thread.start()
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


def require_runtime_ready() -> None:
    state = get_startup_state()
    if state["startup_ready"]:
        return
    detail = state["startup_message"]
    if state["startup_error"]:
        detail = f"{detail}: {state['startup_error']}"
    raise HTTPException(status_code=503, detail=detail)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    state = get_startup_state()
    if state["startup_phase"] == "failed":
        status = "failed"
    elif not state["startup_ready"]:
        status = "initializing"
    elif not state["knowledge_base_ready"]:
        status = "degraded"
    else:
        status = "ok"
    return HealthResponse(
        status=status,
        startup_phase=str(state["startup_phase"]),
        startup_ready=bool(state["startup_ready"]),
        startup_message=str(state["startup_message"]),
        startup_error=state["startup_error"],
        use_local_model=settings.use_local_model,
        model_loaded=bool(getattr(rag_pipeline.llm_client, "is_loaded", False)),
        knowledge_base_ready=bool(state["knowledge_base_ready"]),
        knowledge_base_chunks=int(state["knowledge_base_chunks"]),
        knowledge_base_error=state["knowledge_base_error"],
        local_model_path=str(settings.local_model_path),
        local_lora_adapter_path=(
            str(settings.local_lora_adapter_path) if settings.local_lora_adapter_path else None
        ),
    )


@app.post(f"{settings.api_prefix}/model/test", response_model=ModelTestResponse)
def test_model(request: ModelTestRequest) -> ModelTestResponse:
    require_runtime_ready()
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
    require_runtime_ready()
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
    require_runtime_ready()

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
