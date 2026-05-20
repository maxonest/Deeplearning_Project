"""Global configuration for the local domain QA system."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env."""

    app_name: str = "Local Domain QA System"
    api_prefix: str = "/api"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    faiss_index_dir: Path = PROJECT_ROOT / "embeddings" / "faiss_index"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    dataset_path: Path = PROJECT_ROOT / "data" / "dataset.json"

    default_top_k: int = 5
    max_recent_turns: int = 6
    max_context_chars: int = 6000

    local_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    local_model_endpoint: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
