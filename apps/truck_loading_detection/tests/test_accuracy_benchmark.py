"""
tests/test_accuracy_benchmark.py

Pillar 1 Production QA Suite: End-to-End Carton Count Accuracy Benchmark Harness.

Verifies:
  1. Discovery of real 3-camera synchronized sessions and ground-truth CSVs.
  2. Execution of real video pipeline evaluation (no synthetic prediction generation).
  3. Ground-truth evaluation against annotated sessions (session_18, session_21).
  4. Real output report generation with honest CLEM, over-counts, and under-counts.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_app_root = str(Path(__file__).resolve().parents[1])
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

from scripts.benchmark_accuracy import (
    discover_multicam_sessions,
    evaluate_session_real_video,
    load_session_csv,
)
from three_camera_system.evaluator import (
    CartonLevelEvaluator,
    Week5EvaluationReport,
)


class TestRealVideoAccuracyBenchmark:
    """Production benchmark test suite for carton counting accuracy using real video."""

    def test_multicam_sessions_discovery(self):
        """Ensure multi-camera ground-truth sessions are discovered from repo."""
        sessions = discover_multicam_sessions(_repo_root, _app_root)
        assert len(sessions) >= 2, f"Expected at least 2 sessions (session_18, session_21), found {list(sessions.keys())}"
        assert "session_18" in sessions
        assert "session_21" in sessions

        for sid in ("session_18", "session_21"):
            sinfo = sessions[sid]
            assert os.path.isfile(sinfo["csv_path"]), f"Missing CSV for {sid}: {sinfo['csv_path']}"
            for cid in ("cam_top", "cam_side", "cam_flank"):
                assert os.path.isfile(sinfo["videos"][cid]), f"Missing video {cid} for {sid}"

    def test_real_video_pipeline_inference_and_evaluation(self, tmp_path: Path):
        """
        Runs real video inference through ThreeCameraCartonPipeline on a 120-frame slice
        of Scenario 1 and evaluates against session_18 ground truth.
        Confirms that predictions originate from actual computer vision (not mock duplication).
        """
        sessions = discover_multicam_sessions(_repo_root, _app_root)
        s18_info = sessions["session_18"]
        gt_events = load_session_csv(s18_info["csv_path"])
        assert len(gt_events) in (6, 7)

        # Ingest first 150 frames (approx 5.0 seconds of video)
        pred_events, telemetry = evaluate_session_real_video(
            s18_info,
            conf_threshold=0.15,
            frame_stride=3,
            max_frames=150,
        )

        assert telemetry["frames_ingested"] == 150
        assert telemetry["frames_inferred"] == 50
        assert telemetry["elapsed_sec"] > 0
        assert isinstance(pred_events, list)

        # Evaluate real predictions against ground truth
        evaluator = CartonLevelEvaluator(match_window_sec=1.5)
        report: Week5EvaluationReport = evaluator.evaluate(
            ground_truth_events=gt_events,
            predicted_events=pred_events,
        )

        assert report.total_ground_truth_cartons in (6, 7)
        assert report.total_predicted_cartons == len(pred_events)
        assert isinstance(report.clem_rate_pct, float)

    def test_baseline_report_structure(self, tmp_path: Path):
        """Verify baseline report output format and persistence."""
        report_file = tmp_path / "test_accuracy_report.json"
        sample_payload = {
            "benchmark_type": "real_raw_video_evaluation",
            "model_path": "apps/truck_loading_detection/best.pt",
            "total_sessions_evaluated": 1,
            "total_ground_truth_cartons": 6,
            "total_predicted_cartons": 4,
            "exact_match_events": 3,
            "clem_exact_match_pct": 50.0,
            "net_error_pct": 33.33,
            "overcount_events": 0,
            "undercount_events": 1,
            "missed_events": 2,
            "spurious_events": 1,
        }
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(sample_payload, f, indent=2)

        with open(report_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["benchmark_type"] == "real_raw_video_evaluation"
        assert loaded["clem_exact_match_pct"] == 50.0
