"""
Authentication service — business logic for user authentication.

Architecture decision: Service layer contains business logic, uses repositories
for data access. Services are framework-agnostic and easily testable.
"""

from __future__ import annotations

from uuid import UUID

from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.models import User
from app.repositories.user_repository import UserRepository


class AuthService:
    """
    Authentication business logic.

    Handles user registration, login, token management, and password operations.
    """

    def __init__(self, user_repo: UserRepository) -> None:
        self._user_repo = user_repo

    async def register(
        self,
        *,
        email: str,
        username: str,
        password: str,
    ) -> User:
        """
        Register a new user.

        Args:
            email: User's email (unique)
            username: User's username (unique)
            password: Plaintext password (will be hashed)

        Returns:
            Created User entity

        Raises:
            ConflictError: If email or username already exists
            ValidationError: If password fails validation
        """
        # Validate password strength
        self._validate_password(password)

        # Check for existing email/username
        if await self._user_repo.exists_by_email(email):
            raise ConflictError(message="Email already registered", field="email")

        if await self._user_repo.exists_by_username(username):
            raise ConflictError(message="Username already taken", field="username")

        # Hash password and create user
        password_hash = hash_password(password)
        user = await self._user_repo.create(
            email=email,
            username=username,
            password_hash=password_hash,
        )

        return user

    async def authenticate(self, *, identifier: str, password: str) -> User:
        """
        Authenticate user with email/username and password.

        Args:
            identifier: Email or username
            password: Plaintext password

        Returns:
            User entity if authentication successful

        Raises:
            ValidationError: If credentials are invalid
        """
        user = await self._user_repo.get_by_email_or_username(identifier)
        if not user:
            raise ValidationError(message="Invalid credentials")

        if not user.is_active:
            raise ValidationError(message="Account is deactivated")

        if not verify_password(password, user.password_hash):
            raise ValidationError(message="Invalid credentials")

        return user

    async def create_tokens(self, user: User) -> tuple[str, str]:
        """
        Create access and refresh tokens for a user.

        Args:
            user: Authenticated user

        Returns:
            Tuple of (access_token, refresh_token)
        """
        access_token = create_access_token(
            user_id=str(user.id),
            email=user.email,
            username=user.username,
        )
        refresh_token = create_refresh_token(
            user_id=str(user.id),
            email=user.email,
            username=user.username,
        )
        return access_token, refresh_token

    async def refresh_tokens(self, refresh_token: str) -> tuple[str, str]:
        """
        Create new tokens from a valid refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            Tuple of (new_access_token, new_refresh_token)

        Raises:
            ValidationError: If refresh token is invalid or expired
        """
        try:
            payload = decode_token(refresh_token)
        except Exception as e:
            raise ValidationError(message="Invalid refresh token") from e

        if payload.type != "refresh":
            raise ValidationError(message="Invalid token type")

        user = await self._user_repo.get_by_id(UUID(payload.sub))
        if not user or not user.is_active:
            raise ValidationError(message="User not found or inactive")

        return await self.create_tokens(user)

    async def get_user_by_id(self, user_id: UUID) -> User:
        """
        Get user by ID.

        Raises:
            NotFoundError: If user doesn't exist
        """
        user = await self._user_repo.get_by_id(user_id)
        if not user:
            raise NotFoundError(resource="User", identifier=str(user_id))
        return user

    async def change_password(
        self,
        *,
        user_id: UUID,
        current_password: str,
        new_password: str,
    ) -> User:
        """
        Change user's password.

        Args:
            user_id: User's UUID
            current_password: Current plaintext password
            new_password: New plaintext password

        Returns:
            Updated User entity

        Raises:
            NotFoundError: If user doesn't exist
            ValidationError: If current password is wrong or new password fails validation
        """
        self._validate_password(new_password)

        user = await self.get_user_by_id(user_id)

        if not verify_password(current_password, user.password_hash):
            raise ValidationError(message="Current password is incorrect")

        user.password_hash = hash_password(new_password)
        return await self._user_repo.update(user)

    async def update_profile(
        self,
        *,
        user_id: UUID,
        email: str | None = None,
        username: str | None = None,
    ) -> User:
        """
        Update user profile.

        Args:
            user_id: User's UUID
            email: New email (optional)
            username: New username (optional)

        Returns:
            Updated User entity

        Raises:
            NotFoundError: If user doesn't exist
            ConflictError: If email/username already taken
        """
        user = await self.get_user_by_id(user_id)

        if email is not None and email.lower() != user.email:
            if await self._user_repo.exists_by_email(email):
                raise ConflictError(message="Email already registered", field="email")
            user.email = email.lower()

        if username is not None and username != user.username:
            if await self._user_repo.exists_by_username(username):
                raise ConflictError(message="Username already taken", field="username")
            user.username = username

        return await self._user_repo.update(user)

    def _validate_password(self, password: str) -> None:
        """
        Validate password strength.

        Raises:
            ValidationError: If password doesn't meet requirements
        """
        errors = []
        if len(password) < 8:
            errors.append("Password must be at least 8 characters")
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")

        if errors:
            raise ValidationError(
                message="Password validation failed",
                errors=[{"field": "password", "message": e} for e in errors],
            )