"""Central runtime configuration.

All paths are resolved from the project root by default. On Windows, copy
`.env.example` to `.env` and set `LOCAL_MODEL_PATH` to the local model folder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.embedding_defaults import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Local Domain QA System"
    api_prefix: str = "/api"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    dataset_path: Path = PROJECT_ROOT / "data" / "dataset.json"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    finetune_dataset_path: Path = PROJECT_ROOT / "data" / "finetune" / "sft_dataset_clean.json"

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    faiss_threads: int = 1
    faiss_index_dir: Path = PROJECT_ROOT / "embeddings" / "faiss_index"
    knowledge_base_self_test_query: str = "什么是体适能？"
    retrieval_failure_fallback: bool = True
    chunk_size: int = 600
    chunk_overlap: int = 80
    default_top_k: int = 5

    max_recent_turns: int = 6
    max_context_chars: int = 6000

    use_local_model: bool = False
    local_model_path: Path = PROJECT_ROOT / "models" / "qwen" / "Qwen3.5-9B"
    local_lora_adapter_path: Path | None = None
    local_model_max_new_tokens: int = 2048
    local_model_temperature: float = 0.2
    local_model_top_p: float = 0.9
    local_model_enable_thinking: bool = False
    local_files_only: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            if value.startswith("["):
                return json.loads(value)
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("local_lora_adapter_path", mode="before")
    @classmethod
    def parse_optional_path(cls, value: Any) -> Any:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value

    @field_validator(
        "dataset_path",
        "raw_data_dir",
        "processed_data_dir",
        "finetune_dataset_path",
        "faiss_index_dir",
        "local_model_path",
        "local_lora_adapter_path",
        mode="after",
    )
    @classmethod
    def resolve_relative_paths(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        if value.is_absolute():
            return value
        return PROJECT_ROOT / value


settings = Settings()
