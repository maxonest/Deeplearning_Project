"""API request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from utils.config import settings


class HealthResponse(BaseModel):
    status: str
    use_local_model: bool
    local_model_path: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None
    top_k: int = Field(default=settings.default_top_k, ge=1, le=20)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    documents: list[dict[str, Any]]


class SessionResponse(BaseModel):
    session_id: str


class ConfigResponse(BaseModel):
    app_name: str
    default_top_k: int
    use_local_model: bool
    local_model_path: str
    embedding_model: str
