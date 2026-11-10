"""
tests/test_piece2_security_lockdown.py

Production QA Suite: Piece 2 Security Lockdown & Access Control.

Verifies:
  1. Unauthenticated mutating requests (/api/reset, /process_3angle, /api/tripwires, /api/live_cameras/set)
     strictly return 401 Unauthorized.
  2. Requests with invalid / spoofed API keys return 401 Unauthorized.
  3. Valid API keys via 'Authorization: Bearer <token>' or 'X-API-Key: <token>' are accepted (200).
  4. Safe read-only GET endpoints (e.g. GET /api/3angle/tripwires, /health) remain accessible.
  5. Immutable security audit logs are recorded in EventStore for every mutation (RESET_COUNTS, UPDATE_TRIPWIRES, etc.).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_app_root = str(Path(__file__).resolve().parent.parent)
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

from app import app, get_app_event_store
from three_camera_system.security import CONFIGURED_API_KEYS, DEFAULT_FALLBACK_KEY


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def auth_headers():
    key = next(iter(CONFIGURED_API_KEYS))
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


@pytest.fixture
def x_api_key_headers():
    key = next(iter(CONFIGURED_API_KEYS))
    return {"X-API-Key": key, "Content-Type": "application/json"}


class TestSecurityLockdown:
    """Verifies authentication enforcement and audit logging across mutating routes."""

    def test_unauthenticated_post_returns_401(self, client):
        """Without any token, mutating endpoints must return 401 Unauthorized."""
        endpoints = [
            ("/api/reset", {}),
            ("/process_3angle", {}),
            ("/api/3angle/tripwires", {"cam_top": {"line_pos_frac": 0.5}}),
            ("/api/tripwires", {"cam_side": {"line_pos_frac": 0.45}}),
            ("/api/live_cameras/set", {"cam_top": "0"}),
            ("/api/3angle/set_live_cameras", {"cam_top": "0"}),
            ("/api/live_cameras/disconnect", {}),
            ("/api/set_line_x", {"line_x": 450}),
        ]

        for path, payload in endpoints:
            res = client.post(path, json=payload)
            assert res.status_code == 401, f"Expected 401 on unauthenticated POST {path}, got {res.status_code}"
            data = res.get_json() or {}
            assert data.get("error") == "Unauthorized"
            assert "Valid API key" in data.get("message", "")

    def test_invalid_token_returns_401(self, client):
        """Providing an invalid/spoofed token must be rejected with 401."""
        bad_headers = {"Authorization": "Bearer totally_wrong_secret_key"}
        res = client.post("/api/reset", headers=bad_headers, json={})
        assert res.status_code == 401

        bad_x_header = {"X-API-Key": "invalid_api_key_12345"}
        res = client.post("/api/tripwires", headers=bad_x_header, json={})
        assert res.status_code == 401

    def test_authenticated_post_reset_and_audit_logging(self, client, auth_headers):
        """Authenticated POST /api/reset succeeds and writes immutable audit record."""
        res = client.post("/api/reset", headers=auth_headers, json={})
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("success") is True
        assert data.get("net_count") == 0

        # Verify audit log in EventStore
        store = get_app_event_store()
        assert store is not None
        logs = store.get_security_audit_logs(limit=10)
        assert len(logs) > 0

        latest = logs[0]
        assert latest["action"] == "RESET_COUNTS"
        assert latest["client_ip"] in ("127.0.0.1", "localhost")
        assert "api_user_" in latest["user_identity"]
        assert latest["details"]["endpoint"] == "/api/reset"

    def test_authenticated_post_tripwires_and_audit_logging(self, client, x_api_key_headers):
        """Authenticated POST /api/tripwires updates line configs and logs audit record."""
        payload = {
            "cam_side": {"orientation": "vertical", "line_pos_frac": 0.42},
            "cam_top": {"orientation": "vertical", "line_pos_frac": 0.58},
        }
        res = client.post("/api/tripwires", headers=x_api_key_headers, json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("success") is True

        # Verify audit log recorded UPDATE_TRIPWIRES with payload details
        store = get_app_event_store()
        logs = store.get_security_audit_logs(limit=10)
        tripwire_logs = [l for l in logs if l["action"] == "UPDATE_TRIPWIRES"]
        assert len(tripwire_logs) > 0
        latest_tw = tripwire_logs[0]
        assert latest_tw["action"] == "UPDATE_TRIPWIRES"
        assert "cam_side" in str(latest_tw["details"])

    def test_read_only_get_endpoints_remain_accessible(self, client):
        """Read-only GET requests should not be blocked by mutating auth policy."""
        res_tripwires = client.get("/api/3angle/tripwires")
        assert res_tripwires.status_code == 200

        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.get_json().get("status") in ("healthy", "degraded")
        assert res_health.get_json().get("service") == "truck-loading-detection"

    def test_query_security_audit_logs_endpoint(self, client, auth_headers):
        """GET /api/security/audit_logs returns JSON audit logs for authenticated operators."""
        # Unauthenticated query -> 401
        res_unauth = client.get("/api/security/audit_logs")
        assert res_unauth.status_code == 401

        # Authenticated query -> 200 with logs
        res = client.get("/api/security/audit_logs?limit=5", headers=auth_headers)
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("status") == "ok"
        assert "audit_logs" in data
        assert isinstance(data["audit_logs"], list)
