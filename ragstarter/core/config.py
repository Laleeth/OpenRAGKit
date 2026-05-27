from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "RAG Starter Kit"
    environment: str = "dev"

    data_dir: Path = Path("data")
    uploads_dir: Path = Path("data/uploads")
    chroma_dir: Path = Path("data/chroma")
    sqlite_path: Path = Path("data/app.db")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    max_context_chunks: int = 5

    llm_provider: str = "extractive"
    llm_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434"
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 60.0

    cors_origins: list[str] = ["*"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> list[str]:
        if value is None:
            return ["*"]
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            raw = value.strip()
            if raw in {"", "*"}:
                return ["*"]
            return [v.strip() for v in raw.split(",") if v.strip()]
        return ["*"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
