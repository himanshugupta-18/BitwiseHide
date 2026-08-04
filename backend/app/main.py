"""
FastAPI application factory.

Architecture decisions:
- Lifespan context manager handles startup/shutdown lifecycle — replaces
  deprecated @app.on_event("startup") and @app.on_event("shutdown").
- Global exception handler maps domain exceptions to HTTP responses,
  keeping business logic framework-agnostic.
- CORS middleware is configured from settings — no hardcoded origins.
- Storage directory is auto-created at startup to prevent runtime errors.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.exceptions import (
    BitwiseHideError,
    ConflictError,
    NotFoundError,
    ServiceUnavailableError,
    ValidationError,
)
from app.core.logging import get_logger, setup_logging
from app.db.session import close_db, init_db

logger = get_logger(__name__)

# --- Exception → HTTP status code mapping ---
_EXCEPTION_STATUS_MAP: dict[type[BitwiseHideError], int] = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    ValidationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    ConflictError: status.HTTP_409_CONFLICT,
    ServiceUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifecycle.

    Startup: initialize logging, database pool, file storage directory.
    Shutdown: close database connections, release resources.
    """
    settings = get_settings()

    # --- Startup ---
    setup_logging(settings)
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )

    # Initialize database connection pool
    init_db(settings)

    # Ensure storage directory exists
    storage_path = Path(settings.storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    logger.info("storage_directory_ready", path=str(storage_path.resolve()))

    logger.info("application_started")

    yield

    # --- Shutdown ---
    logger.info("application_shutting_down")
    await close_db()
    logger.info("application_stopped")


def create_app() -> FastAPI:
    """
    Application factory — constructs and configures the FastAPI instance.

    Using a factory function (instead of a module-level app) enables:
    - Testing with different settings
    - Multiple app instances in the same process
    - Clean separation of construction from configuration
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-Powered Visual Password Manager using Image Steganography",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    # --- CORS Middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Global Exception Handlers ---
    @app.exception_handler(BitwiseHideError)
    async def bitwisehide_exception_handler(
        request: Request, exc: BitwiseHideError  # noqa: ARG001
    ) -> ORJSONResponse:
        """Map domain exceptions to HTTP error responses."""
        status_code = _EXCEPTION_STATUS_MAP.get(
            type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR
        )

        logger.error(
            "handled_exception",
            exception_type=type(exc).__name__,
            message=exc.message,
            detail=exc.detail,
            status_code=status_code,
        )

        return ORJSONResponse(
            status_code=status_code,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "detail": exc.detail,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception  # noqa: ARG001
    ) -> ORJSONResponse:
        """
        Catch-all for unexpected exceptions.

        Logs the full traceback but returns a generic message to the client
        to prevent information leakage in production.
        """
        logger.exception("unhandled_exception", error=str(exc))

        message = str(exc) if settings.is_development else "An internal error occurred."

        return ORJSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": message,
                "detail": {},
            },
        )

    # --- Routers ---
    app.include_router(api_v1_router)

    return app


# --- Module-level app instance for uvicorn ---
app = create_app()
