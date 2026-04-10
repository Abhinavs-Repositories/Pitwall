"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM keys
    groq_api_key: str = Field(default="", description="Groq API key for primary LLM")
    google_api_key: str = Field(default="", description="Google AI Studio key for fallback LLM + embeddings")

    # Qdrant
    qdrant_url: str = Field(default="", description="Qdrant Cloud cluster URL")
    qdrant_api_key: str = Field(default="", description="Qdrant API key")
    qdrant_collection: str = Field(default="f1_strategies", description="Qdrant collection name")

    # Storage
    cache_dir: str = Field(default="./data/cache", description="Directory for file-based cache")
    sqlite_db_path: str = Field(default="./data/cache.db", description="SQLite database path")

    # OpenF1
    openf1_base_url: str = Field(default="https://api.openf1.org/v1", description="OpenF1 API base URL")
    openf1_rate_limit_per_second: float = Field(default=3.0, description="Max requests per second to OpenF1")
    openf1_rate_limit_per_minute: int = Field(default=30, description="Max requests per minute to OpenF1")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # LLM model names
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Groq model ID")
    gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini fallback model ID")
    embedding_model: str = Field(default="models/gemini-embedding-001", description="Google embedding model")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
