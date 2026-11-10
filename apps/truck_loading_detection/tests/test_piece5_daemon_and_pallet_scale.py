"""
apps/truck_loading_detection/tests/test_piece5_daemon_and_pallet_scale.py

Production Operationalization, Standalone Daemon, IPC & Pallet Scale Shadow Gate
Piece 5 QA Certification Suite.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict

# Ensure app directory is on path for both relative and absolute test execution
_app_dir = str(Path(__file__).resolve().parent.parent)
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import cv2
import numpy as np
import pytest

from three_camera_system.daemon import DaemonConfig, TruckLoadingProductionDaemon
from three_camera_system.event_store import EventStore
from three_camera_system.live_cameras import CAMERA_IDS, LiveCameraManager
from three_camera_system.pipeline import CrossingWindow, LocalCameraCrossingCandidate, ThreeCameraCartonPipeline
from three_camera_system.shadow_mode import ShadowModeManager


@pytest.fixture
def temp_workspace():
    tmp_dir = tempfile.mkdtemp(prefix="carton_piece5_")
    db_path = os.path.join(tmp_dir, "test_events.db")
    heartbeat_path = os.path.join(tmp_dir, "daemon_heartbeat.json")
    evidence_dir = os.path.join(tmp_dir, "audit_evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    yield {
        "dir": tmp_dir,
        "db_path": db_path,
        "heartbeat_path": heartbeat_path,
        "evidence_dir": evidence_dir,
    }
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_piece5_pallet_scale_shadow_gate_match(temp_workspace):
    """
    Validates Pallet Scale Shadow Gate when optical count matches scale reading:
    N_pallet = round((W_gross - W_tare) / w_carton) == N_optical.
    """
    store = EventStore(db_path=temp_workspace["db_path"])
    sm = ShadowModeManager(event_store=store, session_id="dock_gate_test_01")

    # W_gross = 525kg, W_tare = 25kg, nominal carton = 12.5kg -> N_pallet = 40 cartons
    audit = sm.record_pallet_scale_audit(
        gross_weight_kg=525.0,
        tare_weight_kg=25.0,
        nominal_carton_weight_kg=12.5,
        optical_count=40,
        notes="Truck Dock 1 Pallet 1 verification",
    )

    assert audit["status"] == "MATCH"
    assert audit["is_discrepancy"] is False
    assert audit["discrepancy_delta"] == 0
    assert audit["scale_carton_count"] == 40
    assert audit["optical_carton_count"] == 40
    assert audit["within_tolerance"] is True

    # Verify audit is persisted in SQLite
    history = sm.get_pallet_scale_audits("dock_gate_test_01")
    assert len(history) == 1
    assert history[0]["audit_id"] == audit["audit_id"]
    assert history[0]["status"] == "MATCH"

    # Verify incident log has INFO confirmation
    assert any(inc.incident_type == "PALLET_SCALE_VERIFIED" for inc in sm.incidents)


def test_piece5_pallet_scale_shadow_gate_discrepancy_triggers_alert_and_evidence(temp_workspace):
    """
    Validates Pallet Scale Discrepancy Gate:
    When scale reading diverges from optical count, triggers CRITICAL incident
    and renders synchronized 3-camera audit snapshot collage.
    """
    store = EventStore(db_path=temp_workspace["db_path"])
    sm = ShadowModeManager(event_store=store, session_id="dock_discrepancy_test")

    # Create dummy frames to simulate live camera capture
    custom_frames = {
        "cam_top": np.full((270, 480, 3), 40, dtype=np.uint8),
        "cam_side": np.full((270, 480, 3), 70, dtype=np.uint8),
        "cam_flank": np.full((270, 480, 3), 100, dtype=np.uint8),
    }

    live_cam = LiveCameraManager()
    evidence_path = os.path.join(temp_workspace["evidence_dir"], "scale_mismatch_audit.jpg")

    # Scale: (525 - 25) / 12.5 = 40 cartons; Optical count: 37 cartons -> Delta = -3
    audit = sm.record_pallet_scale_audit(
        gross_weight_kg=525.0,
        tare_weight_kg=25.0,
        nominal_carton_weight_kg=12.5,
        optical_count=37,
        live_camera_manager=live_cam,
        custom_frames=custom_frames,
        notes="Scale discrepancy gate test",
    )

    assert audit["status"] == "UNDERCOUNT_SUSPECTED"
    assert audit["is_discrepancy"] is True
    assert audit["discrepancy_delta"] == -3
    assert audit["evidence_path"] is not None
    assert os.path.exists(audit["evidence_path"])

    # Verify saved image is valid 3-camera collage
    img = cv2.imread(audit["evidence_path"])
    assert img is not None
    assert img.shape[1] == 480 * 3  # 1440 px wide (3 synchronized camera panes)

    # Verify CRITICAL structured incident logged
    disc_incidents = [inc for inc in sm.incidents if inc.incident_type == "COUNT_DISCREPANCY_SCALE_MISMATCH"]
    assert len(disc_incidents) == 1
    assert disc_incidents[0].severity == "CRITICAL"
    assert disc_incidents[0].metadata["discrepancy_delta"] == -3


def test_piece5_pipeline_flush_pending_windows_on_shutdown(temp_workspace):
    """
    Validates that on graceful shutdown, pipeline flushes and commits all uncommitted
    crossing windows to EventStore without dropping in-transit cartons.
    """
    store = EventStore(db_path=temp_workspace["db_path"])
    pipeline = ThreeCameraCartonPipeline(
        model_path="best.pt",
        event_store=store,
        session_id="flush_session_01",
        require_calibrated_fusion=False,
    )

    # Manually inject an uncommitted crossing window with 2 corroborated camera candidates
    win = CrossingWindow(
        window_id=99,
        start_time=10.0,
        last_time=10.5,
        opened_at_time=10.0,
        last_arrival_time=10.5,
        committed=False,
    )
    win.local_events["cam_side"] = [
        LocalCameraCrossingCandidate(
            local_event_id="ev_flush_side_99",
            camera_id="cam_side",
            local_track_id="carton_flush_99",
            timestamp_sec=10.1,
            direction="IN",
            confidence=0.85,
            bbox=(450.0, 200.0, 520.0, 280.0),
            aspect_ratio=1.0,
        )
    ]
    win.local_events["cam_top"] = [
        LocalCameraCrossingCandidate(
            local_event_id="ev_flush_top_99",
            camera_id="cam_top",
            local_track_id="carton_flush_99",
            timestamp_sec=10.2,
            direction="IN",
            confidence=0.80,
            bbox=(460.0, 210.0, 530.0, 290.0),
            aspect_ratio=1.0,
        )
    ]
    pipeline._active_crossing_windows.append(win)

    # Pre-flush assertion: window is uncommitted
    assert any(not w.committed for w in pipeline._active_crossing_windows)
    assert pipeline.confirmed_in == 0

    # Execute graceful flush
    flushed = pipeline.flush_pending_windows(current_time_sec=11.0)
    assert len(flushed) == 1
    assert flushed[0]["global_carton_id"] == "carton_flush_99"
    assert flushed[0]["state"] == "COUNT_COMMITTED"
    assert pipeline.confirmed_in == 1

    # Check durable database commitment
    db_counts = store.get_counts(session_id="flush_session_01")
    assert db_counts["confirmed_in"] == 1


def test_piece5_production_daemon_lifecycle_recovery_and_heartbeat(temp_workspace):
    """
    Validates standalone daemon lifecycle:
    1. Clean startup
    2. Atomic heartbeat JSON updates
    3. Graceful shutdown
    4. Crash/restart recovery from SQLite WAL EventStore
    """
    config = DaemonConfig(
        session_id="dock_daemon_lifecycle",
        model_path="best.pt",
        device="cpu",
        target_fps=15.0,
        db_path=temp_workspace["db_path"],
        heartbeat_path=temp_workspace["heartbeat_path"],
        heartbeat_interval_sec=0.5,
    )

    daemon = TruckLoadingProductionDaemon(config)
    daemon.start(install_signal_handlers=False)

    # Verify heartbeat file is created and valid
    time.sleep(0.6)
    assert os.path.exists(config.heartbeat_path)
    with open(config.heartbeat_path) as f:
        hb = json.load(f)
    assert hb["status"] == "running"
    assert hb["session_id"] == "dock_daemon_lifecycle"
    assert hb["durable_storage"] == "sqlite_wal"

    # Manually commit an event into the session to simulate counting activity
    daemon.event_store.commit_event(
        {
            "global_carton_id": "daemon_carton_101",
            "direction": "IN",
            "carton_count": 1,
            "state": "COUNT_COMMITTED",
            "is_auto_confirmed": 1,
            "participating_cameras": ["cam_side", "cam_top"],
        },
        session_id="dock_daemon_lifecycle",
    )

    # Stop daemon gracefully
    daemon.stop()
    assert not daemon._is_running

    # Verify final heartbeat indicates stopped
    with open(config.heartbeat_path) as f:
        hb_stopped = json.load(f)
    assert hb_stopped["status"] == "stopped"

    # Simulate restart / re-instantiation: new daemon booting on same DB
    restarted_daemon = TruckLoadingProductionDaemon(config)
    restarted_daemon.start(install_signal_handlers=False)

    # Verify recovered count state
    assert restarted_daemon.pipeline.confirmed_in == 1
    restarted_daemon.stop()


def test_piece5_flask_pallet_scale_and_daemon_endpoints(temp_workspace):
    """
    Tests REST API endpoints in app.py:
    - POST /api/3angle/pallet_scale (match & discrepancy)
    - GET /api/3angle/pallet_scale_history
    - GET /api/3angle/daemon_status
    """
    from app import app, get_app_event_store

    client = app.test_client()

    # 1. Test POST /api/3angle/pallet_scale (exact match)
    resp = client.post(
        "/api/3angle/pallet_scale",
        json={
            "session_id": "flask_test_dock",
            "gross_weight_kg": 650.0,
            "tare_weight_kg": 25.0,
            "nominal_carton_weight_kg": 12.5,
            "optical_count": 50,  # (650 - 25)/12.5 = 50
            "notes": "Flask client test exact match",
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["audit"]["status"] == "MATCH"
    assert data["audit"]["is_discrepancy"] is False

    # 2. Test POST /api/3angle/pallet_scale (discrepancy undercount)
    resp_disc = client.post(
        "/api/3angle/pallet_scale",
        json={
            "session_id": "flask_test_dock",
            "gross_weight_kg": 650.0,
            "tare_weight_kg": 25.0,
            "nominal_carton_weight_kg": 12.5,
            "optical_count": 48,  # (650 - 25)/12.5 = 50 != 48
            "notes": "Flask client test discrepancy",
        },
    )
    assert resp_disc.status_code == 200
    data_disc = resp_disc.get_json()
    assert data_disc["success"] is True
    assert data_disc["audit"]["is_discrepancy"] is True
    assert data_disc["audit"]["discrepancy_delta"] == -2

    # 3. Test GET /api/3angle/pallet_scale_history
    resp_hist = client.get("/api/3angle/pallet_scale_history?session_id=flask_test_dock")
    assert resp_hist.status_code == 200
    data_hist = resp_hist.get_json()
    assert data_hist["success"] is True
    assert data_hist["total_audits"] >= 2

    # 4. Test GET /api/3angle/daemon_status
    resp_status = client.get("/api/3angle/daemon_status?session_id=flask_test_dock")
    assert resp_status.status_code == 200
    data_status = resp_status.get_json()
    assert data_status["success"] is True
    assert "live_cameras" in data_status
    assert "shadow_mode" in data_status
    assert data_status["durable_store"]["ready"] is True
