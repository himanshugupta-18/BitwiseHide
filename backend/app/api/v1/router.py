"""
API v1 router — aggregates all endpoint sub-routers.

Architecture decision: Each endpoint module (auth, vault, stego, health)
owns its own APIRouter. This file composes them under a single /api/v1
prefix. Adding a new domain (e.g., /api/v1/admin) requires only one
include_router() call — no changes to main.py.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import auth, health

api_v1_router = APIRouter(prefix="/api/v1")

# --- Register endpoint routers ---
api_v1_router.include_router(health.router)
api_v1_router.include_router(auth.router)

# Future phases will add:
# api_v1_router.include_router(vault.router)
# api_v1_router.include_router(steganography.router)
# api_v1_router.include_router(metrics.router)
