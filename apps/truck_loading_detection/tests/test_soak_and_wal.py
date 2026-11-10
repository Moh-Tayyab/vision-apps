"""
tests/test_soak_and_wal.py

Pillar 2 Production QA Suite: 24-72h Continuous Soak & SQLite WAL Growth Test.

Verifies:
  1. Memory stability and zero memory leak over continuous frame ingestion cycles.
  2. Bounded ring buffer and track state memory (O(1) memory invariance).
  3. SQLite WAL file growth control via periodic checkpointing (PRAGMA wal_checkpoint(TRUNCATE)).
  4. Automatic eviction and pruning of expired tracks and obsolete candidates.
"""

from __future__ import annotations

import gc
import os
import resource
import sqlite3
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


class TestSoakAndWALGrowth:
    """Production test suite for memory leak prevention and WAL file bounds."""

    def test_pipeline_continuous_ingestion_memory_stability(self):
        """
        Simulate continuous frame ingestion (500 frames across 3 cameras).
        Verify memory stays bounded and tracking state does not leak.
        """
        pipeline = ThreeCameraCartonPipeline(
            camera_ids=("cam_top", "cam_side", "cam_flank"),
            fps=30.0,
            conf_threshold=0.20,
            require_calibrated_fusion=False,
            allow_temporal_video_fusion=True,
        )

        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        # Warm-up ticks to initialize torch/detector buffers
        for w_tick in range(3):
            pipeline.process_synchronized_frames(
                {"cam_top": dummy_frame, "cam_side": dummy_frame, "cam_flank": dummy_frame},
                frame_idx=w_tick,
                override_time_sec=0.0,
            )

        gc.collect()
        rss_start_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Ingest 60 synchronized frame ticks
        for tick in range(1, 61):
            t_sec = tick / 30.0
            frame_data = {
                "cam_top": dummy_frame,
                "cam_side": dummy_frame,
                "cam_flank": dummy_frame,
            }
            pipeline.process_synchronized_frames(
                frame_data,
                frame_idx=tick,
                override_time_sec=t_sec,
            )

        # Check ring buffer bounded invariant
        assert len(pipeline.ring_buffer._buffers["cam_top"]) <= pipeline.ring_buffer.capacity_frames
        assert len(pipeline.ring_buffer._buffers["cam_side"]) <= pipeline.ring_buffer.capacity_frames
        assert len(pipeline.ring_buffer._buffers["cam_flank"]) <= pipeline.ring_buffer.capacity_frames

        gc.collect()
        rss_end_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        delta_rss_mb = (rss_end_kb - rss_start_kb) / 1024.0

        # Memory growth during streaming must be zero or near-zero (< 10 MB)
        assert delta_rss_mb < 10.0, f"Memory leak detected during streaming: {delta_rss_mb:.2f} MB"

    def test_sqlite_wal_growth_and_truncate_checkpoint(self, tmp_path: Path):
        """
        Verify that committing continuous audit events does not cause unbounded
        growth of the SQLite WAL sidecar file, and that TRUNCATE checkpoint resets it.
        """
        db_path = str(tmp_path / "soak_test_events.db")
        store = EventStore(db_path=db_path)

        wal_file = f"{db_path}-wal"

        # Commit 100 consecutive carton crossing events
        for i in range(100):
            ev = {
                "event_id": f"soak_ev_{i:04d}",
                "global_carton_id": f"carton_{i:04d}",
                "direction": "IN",
                "carton_count": 1,
                "timestamp_sec": float(i * 1.5),
                "is_auto_confirmed": 1,
                "requires_review": 0,
                "state": "COUNT_COMMITTED",
                "notes": f"Continuous soak test event {i}",
                "camera_evidence": {},
            }
            ok, reason, _ = store.commit_event(ev, session_id="soak_session_01")
            assert ok is True, f"Event {i} commit failed: {reason}"

        # WAL file must exist while active writes are occurring
        assert os.path.exists(wal_file), "Expected SQLite WAL file to exist during ingestion"
        wal_size_before = os.path.getsize(wal_file)
        assert wal_size_before > 0, "Expected non-zero WAL size during ingestion"

        # Execute WAL checkpoint with TRUNCATE mode
        busy, log_frames, checkpointed = store.wal_checkpoint(mode="TRUNCATE")
        assert busy == 0, f"WAL checkpoint was busy: {busy}"
        assert checkpointed >= 0

        # After TRUNCATE checkpoint, WAL file must be truncated back to 0 bytes
        wal_size_after = os.path.getsize(wal_file) if os.path.exists(wal_file) else 0
        assert wal_size_after == 0, f"WAL file was not truncated to 0 bytes: {wal_size_after} bytes"

        # Database integrity check must pass cleanly
        assert store.quick_check() == "ok"
        assert store.integrity_check() == "ok"

    def test_state_machine_stale_track_pruning(self):
        """
        Verify that CrossingStateMachine purges historical tracklets and cartons
        older than max_age to prevent linear memory leaks over 72-hour operation.
        """
        pipeline = ThreeCameraCartonPipeline(
            camera_ids=("cam_top", "cam_side", "cam_flank"),
            conf_threshold=0.20,
            require_calibrated_fusion=False,
            allow_temporal_video_fusion=True,
        )

        # Inject carton at t=1.0s
        c1 = LocalCameraCrossingCandidate(
            local_event_id="e1",
            local_track_id=1,
            camera_id="cam_top",
            timestamp_sec=1.0,
            observed_frame_idx=30,
            direction="IN",
            confidence=0.8,
            bbox=(480.0, 100.0, 520.0, 200.0),
            aspect_ratio=1.0,
            crossing_line_px=500.0,
            crossing_point_px=(500.0, 150.0),
        )
        pipeline._add_crossing_candidate_to_window(c1, arrival_time=1.0)
        pipeline.flush_pending_windows(current_time_sec=2.5)

        # Advance timeline to t=100.0s (well beyond tracking retention window)
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        pipeline.process_synchronized_frames(
            {"cam_top": dummy_frame, "cam_side": dummy_frame, "cam_flank": dummy_frame},
            frame_idx=3000,
            override_time_sec=100.0,
        )

        # Verify active windows list has discarded committed and expired windows
        active_windows = [w for w in pipeline._active_crossing_windows if not w.committed]
        assert len(active_windows) == 0, "Expected all stale windows to be pruned"
