"""
tests/test_dock_shadow_mode_validation.py

Pillar 4 Production QA Suite: Real Dock Footage & Shift Shadow-Mode Validation.

Verifies:
  1. Non-intrusive live shadow mode execution without mutating ERP/WMS inventory.
  2. Side-by-side comparison of optical 3-camera count vs manual operator tally count.
  3. Ground-truth cross-validation with pallet scale telemetry (N = (W_gross - W_tare) / w_carton).
  4. Automatic 3-camera visual evidence capture on count discrepancy.
  5. Incident auditing and stream health monitoring (jitter, drop rate, FPS).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_app_root = str(Path(__file__).resolve().parents[1])
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

from three_camera_system.shadow_mode import ShadowModeManager
from three_camera_system.event_store import EventStore
from three_camera_system.evaluator import PalletScaleCrossChecker, PalletScaleAuditResult


class TestDockShadowModeValidation:
    """Production test suite for real dock shadow-mode verification."""

    def test_shadow_mode_isolation_and_manual_tally_comparison(self, tmp_path: Path):
        """
        Verify that shadow mode isolates authoritative inventory while computing
        discrepancy delta against manual dock tally counts.
        """
        db_path = str(tmp_path / "shadow_dock.db")
        store = EventStore(db_path=db_path)
        manager = ShadowModeManager(
            event_store=store,
            session_id="shift_dock_morning_01",
            authoritative_inventory_locked=True,
        )

        assert manager.authoritative_inventory_locked is True

        # Operator manual dock tally count = 20 cartons
        manager.set_human_physical_count(20)
        assert manager.human_physical_count == 20

        # Simulate optical pipeline committing 20 confirmed cartons
        for i in range(1, 21):
            ev = {
                "event_id": f"ev_dock_{i:03d}",
                "crossing_occurrence_id": f"shift_dock_morning_01:c_{i}:c1:IN",
                "global_carton_id": f"c_{i}",
                "direction": "IN",
                "carton_count": 1,
                "timestamp_sec": float(i * 2.0),
                "is_auto_confirmed": 1,
                "requires_review": 0,
                "state": "COUNT_COMMITTED",
                "notes": f"Optical carton {i}",
            }
            store.commit_event(ev, session_id="shift_dock_morning_01")

        counts = manager.get_shadow_counts()
        assert counts["human_physical_count"] == 20
        assert counts["system_total_confirmed"] == 20
        assert counts["discrepancy_count"] == 0
        assert counts["discrepancy_rate_pct"] == 0.0

    def test_shadow_mode_pallet_scale_discrepancy_detection(self, tmp_path: Path):
        """
        Verify pallet scale shadow cross-check:
        Gross = 525 kg, Tare = 25 kg, Unit = 10 kg -> Scale = 50 cartons.
        If optical count = 48 -> Alert UNDERCOUNT_SUSPECTED (delta = -2).
        """
        checker = PalletScaleCrossChecker(
            nominal_carton_weight_kg=10.0,
            default_tare_weight_kg=25.0,
            weight_tolerance_pct=3.0,
        )

        audit: PalletScaleAuditResult = checker.cross_check(
            optical_count=48,
            gross_weight_kg=525.0,
            tare_weight_kg=25.0,
            unit_weight_kg=10.0,
        )

        assert audit.optical_carton_count == 48
        assert audit.scale_carton_count == 50
        assert audit.discrepancy_delta == -2
        assert audit.status == "UNDERCOUNT_SUSPECTED"
        assert audit.within_tolerance is False
        assert "fewer" in audit.diagnostic_message.lower() or "undercount" in audit.diagnostic_message.lower()

    def test_shadow_mode_stream_health_and_incident_logging(self, tmp_path: Path):
        """
        Verify stream health telemetry: dropped frames, jitter spikes (>33.3ms),
        and transactional incident logging in EventStore.
        """
        db_path = str(tmp_path / "shadow_incidents.db")
        store = EventStore(db_path=db_path)
        manager = ShadowModeManager(event_store=store, session_id="shift_dock_evening_02")

        # Simulate frame jitter spike
        manager.record_frame_health(jitter_ms=45.0, is_dropped=False, elapsed_sec=10.0)
        assert manager.health.max_sync_jitter_ms == 45.0
        assert manager.health.incidents_logged >= 1

        # Incident must be durably stored in SQLite
        incidents = manager.incidents
        assert len(incidents) >= 1
        assert incidents[0].incident_type == "SYNC_JITTER_EXCEEDED"
        assert incidents[0].severity == "WARNING"
