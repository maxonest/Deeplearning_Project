"""FastAPI application for local domain QA."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.memory import ConversationMemory
from backend.rag import RAGPipeline
from backend.schemas import ChatRequest, ChatResponse, ConfigResponse, HealthResponse, SessionResponse
from utils.config import settings


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory = ConversationMemory(
    max_recent_turns=settings.max_recent_turns,
    max_context_chars=settings.max_context_chars,
)
rag_pipeline = RAGPipeline()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        use_local_model=settings.use_local_model,
        local_model_path=str(settings.local_model_path),
    )


@app.get(f"{settings.api_prefix}/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    return ConfigResponse(
        app_name=settings.app_name,
        default_top_k=settings.default_top_k,
        use_local_model=settings.use_local_model,
        local_model_path=str(settings.local_model_path),
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
