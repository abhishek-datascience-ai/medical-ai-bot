from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.app.core.exceptions import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AppSettings(BaseSettings):

    # LLM settings
    llm_provider: Literal["gemini", "local"] = Field(
        default="gemini",
        alias="LLM_PROVIDER",
    )
    llm_mode: Literal["local", "cloud"] = Field(
        default="cloud",
        alias="LLM_MODE",
    )

    # Gemini settings
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(
        default="gemini-3.5-flash-lite",
        alias="GEMINI_MODEL",
    )
    gemini_temperature: float = Field(default=0.2, alias="GEMINI_TEMPERATURE")
    gemini_max_output_tokens: int = Field(
        default=2048,
        alias="80",
    )

    @field_validator("gemini_temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Validate LLM temperature."""

        if not 0 <= value <= 2:
            raise ConfigurationError("Gemini temperature must be between 0 and 2.")

        return value

    @model_validator(mode="after")
    def validate_llm_settings(self) -> AppSettings:
        """Validate LLM settings."""

        if self.llm_mode == "cloud" and self.llm_provider == "gemini":
            if not self.gemini_api_key or self.gemini_api_key == "your-gemini-api-key":
                raise ConfigurationError(
                    "GEMINI_API_KEY must be set when LLM_MODE is cloud."
                )

        return self

    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Medical-AI Bot", alias="APP_NAME")
    app_env: Literal["development", "testing", "staging", "production"] = Field(
        default="development",
        alias="APP_ENV",
    )
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    # Security
    secret_key: str = Field(default="", alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(
        default=60,
        alias="ACCESS_TOKEN_EXPIRE_MINUTES",
    )

    # LLM Provider
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # Qdrant connection
    qdrant_mode: Literal["local", "server", "cloud"] = Field(
        default="local",
        alias="QDRANT_MODE",
    )
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_local_path: Path = Field(
        default=Path("data/vector_store/qdrant"),
        alias="QDRANT_LOCAL_PATH",
    )
    qdrant_collection_name: str = Field(
        default="medical_ai_bot_chunks",
        alias="QDRANT_COLLECTION_NAME",
    )

    # Vector settings
    dense_vector_name: str = Field(default="dense", alias="DENSE_VECTOR_NAME")
    sparse_vector_name: str = Field(default="bm25_sparse", alias="SPARSE_VECTOR_NAME")
    dense_vector_size: int = Field(default=384, alias="DENSE_VECTOR_SIZE")

    # Embedding settings
    dense_embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="DENSE_EMBEDDING_MODEL_NAME",
    )
    sparse_embedding_model_name: str = Field(
        default="Qdrant/bm25",
        alias="SPARSE_EMBEDDING_MODEL_NAME",
    )
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    vector_upload_batch_size: int = Field(
        default=64,
        alias="VECTOR_UPLOAD_BATCH_SIZE",
    )

    # Data paths
    raw_data_dir: Path = Field(
        default=Path("data/raw/mediassist_data"),
        alias="RAW_DATA_DIR",
    )
    processed_data_dir: Path = Field(
        default=Path("data/processed"),
        alias="PROCESSED_DATA_DIR",
    )
    sqlite_db_path: Path = Field(
        default=Path("data/raw/mediassist_data/db/mediassist.db"),
        alias="SQLITE_DB_PATH",
    )

    # Retrieval settings
    dense_top_k: int = Field(default=10, alias="DENSE_TOP_K")
    rerank_top_k: int = Field(default=3, alias="RERANK_TOP_K")

    # Reranking and context settings
    reranker_model_name: str = Field(
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL_NAME",
    )
    context_max_chars: int = Field(default=6000, alias="CONTEXT_MAX_CHARS")

    @field_validator(
        "raw_data_dir",
        "processed_data_dir",
        "sqlite_db_path",
        "qdrant_local_path",
        mode="before",
    )
    @classmethod
    def resolve_project_relative_path(cls, value: str | Path) -> Path:
        """Resolve relative paths from the project root."""

        path_value = Path(value)

        if path_value.is_absolute():
            return path_value

        return PROJECT_ROOT / path_value

    @field_validator("app_port", "qdrant_port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        """Validate TCP port values."""

        if not 1 <= value <= 65535:
            raise ConfigurationError(f"Invalid port value: {value}")

        return value

    @field_validator(
        "dense_top_k",
        "rerank_top_k",
        "dense_vector_size",
        "embedding_batch_size",
        "vector_upload_batch_size",
        "context_max_chars",
    )
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        """Ensure numeric settings are positive integers."""

        if value <= 0:
            raise ConfigurationError("Numeric settings must be greater than zero.")

        return value

    @model_validator(mode="after")
    def validate_retrieval_limits(self) -> AppSettings:
        """Ensure reranking does not request more chunks than initial retrieval."""

        if self.rerank_top_k > self.dense_top_k:
            raise ConfigurationError(
                "RERANK_TOP_K cannot be greater than DENSE_TOP_K."
            )

        return self

    @model_validator(mode="after")
    def validate_qdrant_connection_settings(self) -> AppSettings:
        """Validate Qdrant settings based on selected connection mode."""

        if self.qdrant_mode in {"server", "cloud"} and not self.qdrant_url:
            raise ConfigurationError(
                "QDRANT_URL is required when QDRANT_MODE is server or cloud."
            )

        if self.qdrant_mode == "cloud" and not self.qdrant_api_key:
            raise ConfigurationError(
                "QDRANT_API_KEY is required when QDRANT_MODE is cloud."
            )

        return self

    @model_validator(mode="after")
    def validate_production_security(self) -> AppSettings:
        """Avoid running production with placeholder or missing secrets."""

        if self.app_env == "production":
            if not self.secret_key or self.secret_key == "replace-with-secure-secret-key":
                raise ConfigurationError(
                    "SECRET_KEY must be set securely in production."
                )

            if not self.openai_api_key or self.openai_api_key == "your-openai-api-key":
                raise ConfigurationError(
                    "OPENAI_API_KEY must be set securely in production."
                )

        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings."""

    return AppSettings()