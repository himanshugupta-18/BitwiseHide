"""
Application configuration loaded from environment variables.

Uses Pydantic BaseSettings for validation, type coercion, and .env file support.
Every configurable value flows through this single module — no hardcoded constants
anywhere else in the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Immutable application settings loaded once at startup.

    Values are sourced from environment variables with fallback to .env file.
    All fields are validated and typed — invalid config fails fast at startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "BitwiseHide"
    app_version: str = "0.1.0"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # --- Server ---
    host: str = "0.0.0.0"  # noqa: S104 — bind address, not a secret
    port: int = 8000
    workers: int = 1
    reload: bool = False

    # --- Database ---
    database_url: str = (
        "postgresql+asyncpg://bitwisehide:bitwisehide_secret@localhost:5432/bitwisehide"
    )
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_recycle: int = 3600
    db_echo: bool = False

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- CORS ---
    cors_origins: str | list[str] = "http://localhost:3000,http://localhost:5173"

    # --- File Storage ---
    storage_path: Path = Path("./storage")
    max_image_size_mb: int = 10
    allowed_image_formats: str | list[str] = "png,bmp,tiff"


    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Accept comma-separated string or list for CORS origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("allowed_image_formats", mode="before")
    @classmethod
    def parse_image_formats(cls, v: str | list[str]) -> list[str]:
        """Accept comma-separated string or list for image formats."""
        if isinstance(v, str):
            return [fmt.strip().lower() for fmt in v.split(",") if fmt.strip()]
        return v

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    @property
    def database_url_sync(self) -> str:
        """Synchronous DB URL for Alembic migrations (replaces asyncpg with psycopg2)."""
        return self.database_url.replace("+asyncpg", "+psycopg2")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings singleton.

    Using lru_cache ensures Settings is instantiated exactly once,
    preventing repeated .env file reads and validation on every request.
    """
    return Settings()
