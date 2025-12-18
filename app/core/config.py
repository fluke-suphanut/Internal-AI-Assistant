from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv


class Settings(BaseModel):
    # App
    APP_NAME: str = Field(default="Internal AI Assistant API")
    ENV: Literal["local", "dev", "prod"] = Field(default="local")
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # OpenAI
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key (required)")
    OPENAI_CHAT_MODEL: str = Field(default="gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")

    # Data
    DATA_DIR: Path = Field(default=Path("data"))
    STORAGE_DIR: Path = Field(default=Path("storage"))
    FAISS_INDEX_DIR: Path = Field(default=Path("storage/faiss_index"))

    # Retrieval
    DEFAULT_TOP_K: int = Field(default=5, ge=1, le=20)

    # Optional: request safety
    MAX_QUERY_CHARS: int = Field(default=2000, ge=200, le=20000)

    @property
    def is_openai_configured(self) -> bool:
        return bool(self.OPENAI_API_KEY.strip())


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Loads settings from .env (if present) and environment variables.
    Caches result for reuse.
    """
    global _settings
    if _settings is not None:
        return _settings

    # Load .env from repo root
    load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"), override=False)

    _settings = Settings(
        APP_NAME=os.getenv("APP_NAME", "Internal AI Assistant API"),
        ENV=os.getenv("ENV", "local"),
        LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),

        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
        OPENAI_CHAT_MODEL=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        OPENAI_EMBEDDING_MODEL=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),

        DATA_DIR=Path(os.getenv("DATA_DIR", "data")),
        STORAGE_DIR=Path(os.getenv("STORAGE_DIR", "storage")),
        FAISS_INDEX_DIR=Path(os.getenv("FAISS_INDEX_DIR", "storage/faiss_index")),

        DEFAULT_TOP_K=int(os.getenv("DEFAULT_TOP_K", "5")),
        MAX_QUERY_CHARS=int(os.getenv("MAX_QUERY_CHARS", "2000")),
    )

    # Ensure directories exist
    _settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    _settings.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    return _settings
