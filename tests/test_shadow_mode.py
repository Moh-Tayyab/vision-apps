"""
Unit tests for Live Shadow-Mode Execution & Audit Engine (Phase 1.10).
"""

import json
import pytest
from pathlib import Path
from apps.truck_loading_detection.model_pipeline.shadow_mode import (
    ShadowModeAuditor,
    ShadowEventRecord,
)


def test_shadow_mode_auditor_logging(tmp_path: Path):
    """Verify non-intrusive logging to JSONL audit files."""
    auditor = ShadowModeAuditor(log_dir=tmp_path, model_version="yolo11s_v2.0", engine_profile="pytorch_cpu")

    rec1 = auditor.log_event(
        session_id="session_01",
        camera_id="cam_corridor",
        timestamp_sec=12.5,
        direction="inbound",
        predicted_count=1,
        ground_truth_count=1,
    )

    assert isinstance(rec1, ShadowEventRecord)
    assert rec1.session_id == "session_01"
    assert rec1.is_verified is True

    # Verify JSONL content on disk
    log_file = tmp_path / "shadow_session_01.jsonl"
    assert log_file.exists()

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_uuid"] == rec1.event_uuid
        assert data["predicted_count"] == 1


def test_shadow_mode_progress_tracking(tmp_path: Path):
    """Verify progress tracking towards 30 sessions and 3,000 events."""
    auditor = ShadowModeAuditor(log_dir=tmp_path)

    # Initially empty
    summary0 = auditor.get_progress_summary()
    assert summary0["total_sessions"] == 0
    assert summary0["total_events"] == 0
    assert summary0["is_ready_for_certification"] is False

    # Log 10 events across 2 sessions
    for i in range(5):
        auditor.log_event("session_01", "cam1", float(i), "inbound", 1, ground_truth_count=1)
        auditor.log_event("session_02", "cam1", float(i), "inbound", 1, ground_truth_count=1)

    summary = auditor.get_progress_summary()
    assert summary["total_sessions"] == 2
    assert summary["total_events"] == 10
    assert summary["verified_events"] == 10
    assert summary["is_ready_for_certification"] is False
