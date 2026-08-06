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
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "BitwiseHide"
    app_version: str = "0.1.0"
    app_env: Literal["development", "staging", "production", "testing"] = "development"
    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False

    database_url: str = "postgresql+asyncpg://bitwisehide:bitwisehide_secret@localhost:5432/bitwisehide"
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_recycle: int = 3600
    db_echo: bool = False

    redis_url: str = "redis://localhost:6379/0"

    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    storage_path: Path = Path("./storage")
    max_image_size_mb: int = 10
    allowed_image_formats: list[str] = ["png", "bmp", "tiff"]

    # --- JWT Authentication ---
    jwt_secret_key: str = "dev-secret-change-in-production-min-32-chars"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> list[str]:
        if isinstance(v, list):
            return [str(origin).strip() for origin in v if str(origin).strip()]
        if isinstance(v, str):
            value = v.strip()
            if value.startswith("["):
                import json
                parsed = json.loads(value)
                return [str(origin).strip() for origin in parsed if str(origin).strip()]
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return ["http://localhost:3000", "http://localhost:5173"]

    @field_validator("allowed_image_formats", mode="before")
    @classmethod
    def parse_image_formats(cls, v: object) -> list[str]:
        if isinstance(v, list):
            return [str(fmt).strip().lower() for fmt in v if str(fmt).strip()]
        if isinstance(v, str):
            value = v.strip()
            if value.startswith("["):
                import json
                parsed = json.loads(value)
                return [str(fmt).strip().lower() for fmt in parsed if str(fmt).strip()]
            return [fmt.strip().lower() for fmt in value.split(",") if fmt.strip()]
        return ["png", "bmp", "tiff"]

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
        return self.database_url.replace("+asyncpg", "+psycopg2")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()