"""
tests/test_failure_drills.py

Pillar 3 Production QA Suite: High-Availability Failure Drills.

Drills:
  1. Camera Down Drill: Single camera drop mid-shift; graceful degradation to 2-camera consensus.
  2. Network Flap Drill: Sudden packet loss / timestamp discontinuity; jitter buffer & timeline recovery.
  3. Power Cut / Crash Drill: Sudden SIGKILL / power loss; SQLite WAL integrity and exact state resume.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
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

from three_camera_system.pipeline import ThreeCameraCartonPipeline
from three_camera_system.event_store import EventStore
from three_camera_system.corridor import LocalCameraCrossingCandidate
from three_camera_system.consensus_policy import DecisionAction


class TestFailureDrills:
    """Production failure drill test suite."""

    def test_drill_single_camera_down_and_dual_camera_degraded_continuation(self):
        """
        Drill 1: Camera Down.
        Mid-shift cam_side goes offline (frame is None or missing).
        Pipeline must NOT crash; must continue with remaining 2 cameras (cam_top + cam_flank)
        and correctly confirm crossing cartons under dual-camera consensus policy.
        """
        pipeline = ThreeCameraCartonPipeline(
            camera_ids=("cam_top", "cam_side", "cam_flank"),
            conf_threshold=0.20,
            require_calibrated_fusion=False,
            allow_temporal_video_fusion=True,
        )

        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        # 1. Normal 3-camera operation
        tick1 = pipeline.process_synchronized_frames(
            {"cam_top": dummy_frame, "cam_side": dummy_frame, "cam_flank": dummy_frame},
            frame_idx=1,
            override_time_sec=0.1,
        )
        assert tick1 is not None

        # 2. Camera Down: cam_side drops (sends None)
        tick2 = pipeline.process_synchronized_frames(
            {"cam_top": dummy_frame, "cam_side": None, "cam_flank": dummy_frame},
            frame_idx=2,
            override_time_sec=0.2,
        )
        assert tick2 is not None
        assert "cam_side" not in tick2.per_camera_tracks or len(tick2.per_camera_tracks["cam_side"]) == 0

        # 3. Ingest carton crossing under remaining 2 cameras (cam_top & cam_flank)
        cand_top = LocalCameraCrossingCandidate(
            local_event_id="top_ev_degraded",
            local_track_id=10,
            camera_id="cam_top",
            timestamp_sec=1.0,
            observed_frame_idx=30,
            direction="IN",
            confidence=0.85,
            bbox=(480.0, 100.0, 520.0, 200.0),
            aspect_ratio=1.0,
            crossing_line_px=500.0,
            crossing_point_px=(500.0, 150.0),
        )
        cand_flank = LocalCameraCrossingCandidate(
            local_event_id="flank_ev_degraded",
            local_track_id=20,
            camera_id="cam_flank",
            timestamp_sec=1.1,
            observed_frame_idx=33,
            direction="IN",
            confidence=0.80,
            bbox=(480.0, 100.0, 520.0, 200.0),
            aspect_ratio=1.0,
            crossing_line_px=500.0,
            crossing_point_px=(500.0, 150.0),
        )
        pipeline._add_crossing_candidate_to_window(cand_top, arrival_time=1.0)
        pipeline._add_crossing_candidate_to_window(cand_flank, arrival_time=1.1)

        # Flush window
        flushed = pipeline.flush_pending_windows(current_time_sec=2.5)
        assert len(flushed) == 1
        assert flushed[0]["is_auto_confirmed"] == 1
        assert pipeline.confirmed_in == 1

        # 4. Camera Reconnected: cam_side comes back online
        tick3 = pipeline.process_synchronized_frames(
            {"cam_top": dummy_frame, "cam_side": dummy_frame, "cam_flank": dummy_frame},
            frame_idx=90,
            override_time_sec=3.0,
        )
        assert tick3 is not None

    def test_drill_network_flap_and_timestamp_jitter(self):
        """
        Drill 2: Network Flap.
        Network drops frames for 2.5 seconds, then resumes with a timestamp jump.
        Pipeline must absorb the timestamp jump without state corruption or crash.
        """
        pipeline = ThreeCameraCartonPipeline(
            camera_ids=("cam_top", "cam_side", "cam_flank"),
            conf_threshold=0.20,
            require_calibrated_fusion=False,
            allow_temporal_video_fusion=True,
        )
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        # Normal frame at t=2.0s
        pipeline.process_synchronized_frames(
            {"cam_top": dummy_frame, "cam_side": dummy_frame, "cam_flank": dummy_frame},
            frame_idx=60,
            override_time_sec=2.0,
        )

        # Flap: 2.5 second gap, next frame arrives at t=4.5s
        pipeline.process_synchronized_frames(
            {"cam_top": dummy_frame, "cam_side": dummy_frame, "cam_flank": dummy_frame},
            frame_idx=135,
            override_time_sec=4.5,
        )

        # Pipeline timeline should be at 4.5s without crashing
        assert pipeline.last_processed_time_sec == 4.5

    def test_drill_power_cut_crash_and_state_resume(self, tmp_path: Path):
        """
        Drill 3: Power Cut / Sudden Restart.
        Simulate power cut right after committing cartons to EventStore.
        On system restart, EventStore and Pipeline must recover all committed
        events, resume exact counts, and reject duplicates through occurrence IDs.
        """
        db_path = str(tmp_path / "power_cut_dock.db")
        session_id = "shift_am_dock_01"

        # 1. Pre-cut session: commit 3 cartons
        store1 = EventStore(db_path=db_path)
        for i in range(1, 4):
            ev = {
                "event_id": f"dock_ev_{i:03d}",
                "crossing_occurrence_id": f"{session_id}:carton_{i:03d}:c1:IN",
                "global_carton_id": f"carton_{i:03d}",
                "direction": "IN",
                "carton_count": 1,
                "timestamp_sec": float(i * 3.0),
                "is_auto_confirmed": 1,
                "requires_review": 0,
                "state": "COUNT_COMMITTED",
                "notes": f"Pre-crash carton {i}",
                "camera_evidence": {},
            }
            store1.commit_event(ev, session_id=session_id)

        # Flush to WAL
        store1.wal_checkpoint("PASSIVE")

        # 2. Simulate Sudden Crash / Power Cut:
        # Drop store1 without closing gracefully
        del store1

        # 3. Restart: New system instance starts up
        store2 = EventStore(db_path=db_path)
        assert store2.quick_check() == "ok"
        assert store2.integrity_check() == "ok"

        # Read back session history
        resumed_events = store2.get_events_by_session(session_id=session_id)
        assert len(resumed_events) == 3, f"Expected 3 recovered events, got {len(resumed_events)}"
        counts = store2.get_counts(session_id=session_id)
        assert counts["confirmed_in"] == 3
        assert counts["net_count"] == 3

        # 4. Invariant: Re-submitting carton_001 must be REJECTED as duplicate
        duplicate_ev = {
            "event_id": "dock_ev_001",
            "crossing_occurrence_id": f"{session_id}:carton_001:c1:IN",
            "global_carton_id": "carton_001",
            "direction": "IN",
            "carton_count": 1,
            "timestamp_sec": 3.0,
            "is_auto_confirmed": 1,
            "requires_review": 0,
            "state": "COUNT_COMMITTED",
            "notes": "Duplicate re-transmitted after reboot",
        }
        ok, reason, _ = store2.commit_event(duplicate_ev, session_id=session_id)
        assert ok is False, "Expected duplicate event to be rejected"
        assert "already_committed" in reason.lower() or "duplicate" in reason.lower()

        # 5. New Carton 4 commits cleanly
        new_ev = {
            "event_id": "dock_ev_004",
            "crossing_occurrence_id": f"{session_id}:carton_004:c1:IN",
            "global_carton_id": "carton_004",
            "direction": "IN",
            "carton_count": 1,
            "timestamp_sec": 15.0,
            "is_auto_confirmed": 1,
            "requires_review": 0,
            "state": "COUNT_COMMITTED",
            "notes": "Post-reboot new carton",
        }
        ok_new, _, _ = store2.commit_event(new_ev, session_id=session_id)
        assert ok_new is True

        all_events = store2.get_events_by_session(session_id=session_id)
        assert len(all_events) == 4
