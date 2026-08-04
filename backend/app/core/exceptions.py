"""
Custom exception hierarchy for BitwiseHide.

Architecture decision: Domain-specific exceptions decouple business logic from HTTP
concerns. Services raise semantic exceptions (NotFoundError, ValidationError);
the global exception handler in main.py maps them to HTTP status codes.

This ensures services remain framework-agnostic and testable without FastAPI.
"""

from __future__ import annotations

from typing import Any


class BitwiseHideError(Exception):
    """
    Base exception for all application errors.

    All custom exceptions inherit from this, enabling catch-all
    handling at the API boundary while preserving specific types
    for targeted recovery in service logic.
    """

    def __init__(
        self,
        message: str = "An unexpected error occurred.",
        *,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.detail = detail or {}
        super().__init__(self.message)


class NotFoundError(BitwiseHideError):
    """Raised when a requested resource does not exist."""

    def __init__(
        self,
        resource: str = "Resource",
        identifier: str | None = None,
    ) -> None:
        msg = f"{resource} not found"
        if identifier:
            msg = f"{resource} with id '{identifier}' not found"
        super().__init__(message=msg, detail={"resource": resource, "identifier": identifier})


class ValidationError(BitwiseHideError):
    """Raised when input data fails domain validation rules."""

    def __init__(
        self,
        message: str = "Validation failed.",
        *,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message=message, detail={"errors": errors or []})


class ConflictError(BitwiseHideError):
    """Raised when an operation conflicts with existing state (e.g., duplicate email)."""

    def __init__(
        self,
        message: str = "Resource already exists.",
        *,
        field: str | None = None,
    ) -> None:
        super().__init__(message=message, detail={"field": field})


class DatabaseError(BitwiseHideError):
    """Raised when a database operation fails unexpectedly."""

    def __init__(self, message: str = "A database error occurred.") -> None:
        super().__init__(message=message)


class StorageError(BitwiseHideError):
    """Raised when file storage operations fail (read, write, delete)."""

    def __init__(self, message: str = "A file storage error occurred.") -> None:
        super().__init__(message=message)


class ServiceUnavailableError(BitwiseHideError):
    """Raised when a required external service (DB, Redis, AI) is unreachable."""

    def __init__(self, service: str = "External service") -> None:
        super().__init__(
            message=f"{service} is currently unavailable.",
            detail={"service": service},
        )
