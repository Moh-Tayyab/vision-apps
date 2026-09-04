"""Security & Role-Based Access Control (RBAC) for Face Authorization.

Protects sensitive endpoints (enrollment, deletion, threshold tuning, camera control)
via API Key / Bearer tokens while allowing open stream viewing for client dashboards.
"""

from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Query, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

API_KEY_NAME = "X-API-Key"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
BEARER_AUTH = HTTPBearer(auto_error=False)

# Configurable master API Key (defaults to secure dev token if not specified)
MASTER_API_KEY = os.getenv("FACE_AUTH_API_KEY", "face-auth-dev-key-2026")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() in ("true", "1", "yes")


def verify_admin_access(
    header_key: Optional[str] = Security(API_KEY_HEADER),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(BEARER_AUTH),
    query_key: Optional[str] = Query(None, alias="api_key"),
) -> bool:
    """Validate API Key or Bearer Token for administrative write endpoints."""
    if not REQUIRE_AUTH:
        return True  # Dev / Local default

    token = None
    if header_key:
        token = header_key
    elif bearer_token:
        token = bearer_token.credentials
    elif query_key:
        token = query_key

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key or Bearer Token. Provide via 'X-API-Key' header or '?api_key=' param.",
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(token.strip(), MASTER_API_KEY.strip()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key or authorization token.",
        )

    return True
