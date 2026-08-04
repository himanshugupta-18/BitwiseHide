"""
Health check endpoint.

Reports application status, database connectivity, and service metadata.
Used by Docker HEALTHCHECK, load balancers, and monitoring systems.

Returns degraded status if any dependency is unhealthy — allows partial
availability while signaling that something needs attention.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, status
from sqlalchemy import text

from app.core.config import get_settings
from app.core.dependencies import DatabaseDep
from app.core.logging import get_logger

router = APIRouter(tags=["Health"])
logger = get_logger(__name__)


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Service health check",
    response_description="Health status of the application and its dependencies.",
)
async def health_check(db: DatabaseDep) -> dict[str, Any]:
    """
    Check the health of the application and all critical dependencies.

    Returns:
        - **status**: "healthy" | "degraded" | "unhealthy"
        - **version**: Application version from settings
        - **checks**: Individual dependency health results
    """
    settings = get_settings()
    checks: dict[str, dict[str, Any]] = {}

    # --- Database connectivity ---
    try:
        result = await db.execute(text("SELECT 1"))
        result.scalar_one()
        checks["database"] = {"status": "healthy", "latency_ms": None}
    except Exception as exc:
        logger.error("health_check_db_failed", error=str(exc))
        checks["database"] = {"status": "unhealthy", "error": str(exc)}

    # --- Aggregate status ---
    all_healthy = all(check["status"] == "healthy" for check in checks.values())
    overall_status = "healthy" if all_healthy else "degraded"

    return {
        "status": overall_status,
        "version": settings.app_version,
        "environment": settings.app_env,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "checks": checks,
    }
