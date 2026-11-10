"""
Security & Access Control Enforcement (Piece 2 Architecture).

Provides API Key and Bearer Token validation, timing-attack-safe comparison,
and immutable security audit logging for mutating endpoints.
"""

from __future__ import annotations

import functools
import hmac
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set

from flask import jsonify, request

logger = logging.getLogger("three_camera_system.security")

# Primary secret configured via environment variable
DEFAULT_FALLBACK_KEY = "carton_secret_token_production"
_CONFIGURED_KEYS_STR = os.environ.get(
    "CARTON_API_KEY",
    os.environ.get("API_KEY", DEFAULT_FALLBACK_KEY),
)

# Support multiple comma-separated keys (e.g., operator key, automation key)
CONFIGURED_API_KEYS: Set[str] = {
    k.strip() for k in _CONFIGURED_KEYS_STR.split(",") if k.strip()
}


def get_client_ip() -> str:
    """Extracts client IP, respecting trusted proxy headers."""
    if request.headers.get("X-Forwarded-For"):
        return request.headers["X-Forwarded-For"].split(",")[0].strip()
    if request.headers.get("X-Real-IP"):
        return request.headers["X-Real-IP"].strip()
    return request.remote_addr or "127.0.0.1"


def extract_token_from_request() -> Optional[str]:
    """
    Extracts authentication credentials from:
    1. Header: Authorization: Bearer <token>
    2. Header: X-API-Key: <token>
    3. Query parameter: ?api_key=<token> or ?token=<token>
    4. Form/JSON payload: api_key or token
    """
    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header:
        parts = auth_header.split(None, 1)
        if len(parts) == 2 and parts[0].lower() in ("bearer", "token", "apikey"):
            return parts[1].strip()
        if len(parts) == 1:
            return parts[0].strip()

    x_api_key = request.headers.get("X-API-Key", "").strip()
    if x_api_key:
        return x_api_key

    token_param = request.args.get("api_key") or request.args.get("token")
    if token_param:
        return token_param.strip()

    if request.is_json:
        try:
            body = request.get_json(silent=True) or {}
            if isinstance(body, dict):
                body_token = body.get("api_key") or body.get("token")
                if body_token:
                    return str(body_token).strip()
        except Exception:
            pass

    if request.form:
        form_token = request.form.get("api_key") or request.form.get("token")
        if form_token:
            return str(form_token).strip()

    return None


def is_valid_token(token: Optional[str]) -> bool:
    """Timing-attack-resistant constant-time token comparison."""
    if not token:
        return False
    for valid_key in CONFIGURED_API_KEYS:
        if hmac.compare_digest(token.encode("utf-8"), valid_key.encode("utf-8")):
            return True
    return False


def require_api_key(action_name: Optional[str] = None, mutating_only: bool = False):
    """
    Flask route decorator enforcing API key / Bearer token authentication.
    Emits 401 Unauthorized if missing/invalid, and creates a security audit log on success.
    
    If mutating_only is True, GET/HEAD requests are permitted without authentication.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if mutating_only and request.method in ("GET", "HEAD", "OPTIONS"):
                return fn(*args, **kwargs)

            token = extract_token_from_request()
            client_ip = get_client_ip()

            if not is_valid_token(token):
                logger.warning(
                    "[SECURITY ALERT] Unauthorized access attempt to %s (%s) from IP %s",
                    request.path,
                    request.method,
                    client_ip,
                )
                return jsonify({
                    "status": "error",
                    "error": "Unauthorized",
                    "message": "Valid API key or Bearer token required for mutating operations",
                    "client_ip": client_ip,
                }), 401

            # User identity tag (masked key preview)
            user_id = f"api_user_{token[:4]}***" if token else "api_user"

            # Execute the endpoint
            response = fn(*args, **kwargs)

            # Record audit log if action_name specified and request succeeded (or returned dict/response)
            if action_name:
                try:
                    from app import get_app_event_store
                    store = get_app_event_store()
                    payload = {}
                    if request.is_json:
                        payload = request.get_json(silent=True) or {}
                    elif request.form:
                        payload = dict(request.form)

                    # Redact any sensitive credentials from audit payload
                    sanitized_payload = {
                        k: v for k, v in payload.items()
                        if k.lower() not in ("api_key", "token", "password", "secret")
                    }

                    if store:
                        store.record_security_audit(
                            action=action_name,
                            client_ip=client_ip,
                            user_identity=user_id,
                            details={
                                "endpoint": request.path,
                                "method": request.method,
                                "payload": sanitized_payload,
                            },
                        )
                    logger.info(
                        "[SECURITY AUDIT] Action '%s' executed on %s by %s from IP %s",
                        action_name,
                        request.path,
                        user_id,
                        client_ip,
                    )
                except Exception as exc:
                    logger.error("Failed to record security audit log: %s", exc)

            return response
        return wrapper
    return decorator
