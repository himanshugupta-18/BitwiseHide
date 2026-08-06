"""
Authentication API tests.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


class TestRegister:
    """Test user registration endpoint."""

    async def test_register_success(self, client: AsyncClient):
        """Test successful user registration."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "username": "newuser",
                "password": "ValidPass123!",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "user" in data
        assert "tokens" in data
        assert data["user"]["email"] == "newuser@example.com"
        assert data["user"]["username"] == "newuser"
        assert data["user"]["is_active"] is True
        assert data["user"]["is_verified"] is False
        assert "access_token" in data["tokens"]
        assert "refresh_token" in data["tokens"]
        assert data["tokens"]["token_type"] == "bearer"

    async def test_register_duplicate_email(self, client: AsyncClient, test_user):
        """Test registration with duplicate email fails."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "username": "differentuser",
                "password": "ValidPass123!",
            },
        )
        assert response.status_code == 409
        assert "Email already registered" in response.json()["message"]

    async def test_register_duplicate_username(self, client: AsyncClient, test_user):
        """Test registration with duplicate username fails."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "different@example.com",
                "username": "testuser",
                "password": "ValidPass123!",
            },
        )
        assert response.status_code == 409
        assert "Username already taken" in response.json()["message"]

    async def test_register_invalid_email(self, client: AsyncClient):
        """Test registration with invalid email fails."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "invalid-email",
                "username": "newuser",
                "password": "ValidPass123!",
            },
        )
        assert response.status_code == 422

    async def test_register_short_username(self, client: AsyncClient):
        """Test registration with short username fails."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "username": "ab",
                "password": "ValidPass123!",
            },
        )
        assert response.status_code == 422

    async def test_register_weak_password(self, client: AsyncClient):
        """Test registration with weak password fails."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "username": "newuser",
                "password": "weak",
            },
        )
        assert response.status_code == 422


class TestLogin:
    """Test user login endpoint."""

    async def test_login_with_email(self, client: AsyncClient, test_user):
        """Test login with email succeeds."""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "identifier": "test@example.com",
                "password": "TestPass123!",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["email"] == "test@example.com"
        assert "access_token" in data["tokens"]
        assert "refresh_token" in data["tokens"]

    async def test_login_with_username(self, client: AsyncClient, test_user):
        """Test login with username succeeds."""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "identifier": "testuser",
                "password": "TestPass123!",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["username"] == "testuser"

    async def test_login_wrong_password(self, client: AsyncClient, test_user):
        """Test login with wrong password fails."""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "identifier": "test@example.com",
                "password": "WrongPass123!",
            },
        )
        assert response.status_code == 400
        assert "Invalid credentials" in response.json()["message"]

    async def test_login_nonexistent_user(self, client: AsyncClient):
        """Test login with nonexistent user fails."""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "identifier": "nonexistent@example.com",
                "password": "TestPass123!",
            },
        )
        assert response.status_code == 400
        assert "Invalid credentials" in response.json()["message"]


class TestRefreshToken:
    """Test token refresh endpoint."""

    async def test_refresh_success(self, client: AsyncClient, test_user):
        """Test successful token refresh."""
        # First login to get tokens
        login_response = await client.post(
            "/api/v1/auth/login",
            json={
                "identifier": "test@example.com",
                "password": "TestPass123!",
            },
        )
        assert login_response.status_code == 200
        refresh_token = login_response.json()["tokens"]["refresh_token"]

        # Refresh tokens
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        # New tokens should be different
        assert data["access_token"] != login_response.json()["tokens"]["access_token"]
        assert data["refresh_token"] != refresh_token

    async def test_refresh_invalid_token(self, client: AsyncClient):
        """Test refresh with invalid token fails."""
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid.token.here"},
        )
        assert response.status_code == 400
        assert "Invalid refresh token" in response.json()["message"]

    async def test_refresh_with_access_token(self, client: AsyncClient, test_user):
        """Test refresh with access token instead of refresh token fails."""
        login_response = await client.post(
            "/api/v1/auth/login",
            json={
                "identifier": "test@example.com",
                "password": "TestPass123!",
            },
        )
        access_token = login_response.json()["tokens"]["access_token"]

        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": access_token},
        )
        assert response.status_code == 400
        assert "Invalid token type" in response.json()["message"]


class TestGetCurrentUser:
    """Test get current user endpoint."""

    async def test_me_success(self, client: AsyncClient, test_user, auth_headers):
        """Test getting current user profile."""
        response = await client.get("/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
        assert data["is_active"] is True

    async def test_me_no_token(self, client: AsyncClient):
        """Test getting current user without token fails."""
        response = await client.get("/api/v1/auth/me")
        assert response.status_code == 401

    async def test_me_invalid_token(self, client: AsyncClient):
        """Test getting current user with invalid token fails."""
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert response.status_code == 401


class TestChangePassword:
    """Test change password endpoint."""

    async def test_change_password_success(self, client: AsyncClient, test_user, auth_headers):
        """Test successful password change."""
        response = await client.post(
            "/api/v1/auth/change-password",
            headers=auth_headers,
            json={
                "current_password": "TestPass123!",
                "new_password": "NewPass456!",
            },
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Password changed successfully"

        # Verify new password works
        login_response = await client.post(
            "/api/v1/auth/login",
            json={
                "identifier": "test@example.com",
                "password": "NewPass456!",
            },
        )
        assert login_response.status_code == 200

    async def test_change_password_wrong_current(self, client: AsyncClient, test_user, auth_headers):
        """Test password change with wrong current password fails."""
        response = await client.post(
            "/api/v1/auth/change-password",
            headers=auth_headers,
            json={
                "current_password": "WrongPass123!",
                "new_password": "NewPass456!",
            },
        )
        assert response.status_code == 400
        assert "Current password is incorrect" in response.json()["message"]

    async def test_change_password_weak_new(self, client: AsyncClient, test_user, auth_headers):
        """Test password change with weak new password fails."""
        response = await client.post(
            "/api/v1/auth/change-password",
            headers=auth_headers,
            json={
                "current_password": "TestPass123!",
                "new_password": "weak",
            },
        )
        assert response.status_code == 422


class TestUpdateProfile:
    """Test update profile endpoint."""

    async def test_update_username(self, client: AsyncClient, test_user, auth_headers):
        """Test updating username."""
        response = await client.patch(
            "/api/v1/auth/me",
            headers=auth_headers,
            json={"username": "newusername"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "newusername"
        assert data["email"] == "test@example.com"

    async def test_update_email(self, client: AsyncClient, test_user, auth_headers):
        """Test updating email."""
        response = await client.patch(
            "/api/v1/auth/me",
            headers=auth_headers,
            json={"email": "newemail@example.com"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "newemail@example.com"
        assert data["username"] == "testuser"

    async def test_update_duplicate_username(self, client: AsyncClient, test_user, auth_headers, test_session):
        """Test updating to existing username fails."""
        from app.core.security import hash_password
        from app.models import User

        # Create another user
        other_user = User(
            email="other@example.com",
            username="otheruser",
            password_hash=hash_password("TestPass123!"),
        )
        test_session.add(other_user)
        await test_session.commit()

        response = await client.patch(
            "/api/v1/auth/me",
            headers=auth_headers,
            json={"username": "otheruser"},
        )
        assert response.status_code == 409
        assert "Username already taken" in response.json()["message"]

    async def test_update_invalid_username(self, client: AsyncClient, test_user, auth_headers):
        """Test updating with invalid username format fails."""
        response = await client.patch(
            "/api/v1/auth/me",
            headers=auth_headers,
            json={"username": "invalid username!"},
        )
        assert response.status_code == 422