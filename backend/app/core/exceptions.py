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


class InvalidCredentialsError(BitwiseHideError):
    """
    Raised when authentication fails (bad credentials, invalid/expired token,
    wrong current password, deactivated account).

    Distinct from ValidationError so the API can return 400 instead of 422 —
    a wrong password is a request that is valid in shape but failed
    authentication, not malformed input.
    """

    def __init__(self, message: str = "Invalid credentials.") -> None:
        super().__init__(message=message)


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


class EncryptionError(BitwiseHideError):
    """
    Raised when encryption/decryption fails or an encrypted payload is
    invalid, malformed, or has been tampered with.

    The encryption service FAILS CLOSED: it never returns corrupted plaintext.
    Any failure — wrong password, modified ciphertext/nonce/salt/version,
    malformed base64, unsupported payload version, or out-of-bounds KDF
    parameters — surfaces as this exception.
    """

    def __init__(self, message: str = "Encryption/decryption failed.") -> None:
        super().__init__(message=message)


class SteganographyError(BitwiseHideError):
    """
    Raised when embedding or extracting a steganographic payload fails.

    Covers invalid/malformed/truncated images, images carrying no valid
    BitwiseHide payload, insufficient image capacity, and payload lengths that
    exceed the image's capacity. The steganography layer FAILS CLOSED: it never
    silently truncates a payload or returns partial/corrupted data.
    """

    def __init__(self, message: str = "Steganography operation failed.") -> None:
        super().__init__(message=message)


class ServiceUnavailableError(BitwiseHideError):
    """Raised when a required external service (DB, Redis, AI) is unreachable."""

    def __init__(self, service: str = "External service") -> None:
        super().__init__(
            message=f"{service} is currently unavailable.",
            detail={"service": service},
        )
