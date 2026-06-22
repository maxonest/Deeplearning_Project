"""API request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from utils.config import settings


class HealthResponse(BaseModel):
    status: str
    startup_phase: str
    startup_ready: bool
    startup_message: str
    startup_error: str | None
    use_local_model: bool
    model_loaded: bool
    knowledge_base_ready: bool
    knowledge_base_chunks: int
    knowledge_base_error: str | None
    local_model_path: str
    local_lora_adapter_path: str | None


class ModelTestRequest(BaseModel):
    prompt: str = Field(default="你好，请用一句话介绍你自己。", min_length=1)
    enable_thinking: bool = Field(default=settings.local_model_enable_thinking)


class ModelTestResponse(BaseModel):
    answer: str
    model_loaded: bool
    local_model_path: str
    local_lora_adapter_path: str | None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None
    top_k: int = Field(default=settings.default_top_k, ge=1, le=20)
    enable_thinking: bool = Field(default=settings.local_model_enable_thinking)


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
    local_lora_adapter_path: str | None
    embedding_model: str
    embedding_query_prompt_name: str
