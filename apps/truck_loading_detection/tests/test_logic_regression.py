"""
tests/test_logic_regression.py

Mathematical & Evaluator Logic Regression Suite.

Verifies the mathematical correctness and edge-case handling of CartonLevelEvaluator,
ConsensusPolicy, and PhysicalCarton deduplication logic on controlled, synthetic vectors:
  1. Exact match evaluations with known offset deltas.
  2. Over-count and under-count metric tracking on stacked cartons.
  3. Visual occlusion coasting and single-camera vs dual-camera review thresholds.
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

from three_camera_system.evaluator import (
    CartonLevelEvaluator,
    PalletScaleCrossChecker,
    Week5EvaluationReport,
)
from three_camera_system.pipeline import ThreeCameraCartonPipeline
from three_camera_system.corridor import LocalCameraCrossingCandidate
from three_camera_system.consensus_policy import DecisionAction


def load_ground_truth_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Loads ground-truth crossing events from standardized CSV."""
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(_repo_root, csv_path)
    if not os.path.isfile(csv_path):
        return []

    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                "event_id": row["event_id"],
                "timestamp_sec": float(row["event_timestamp"]),
                "direction": row["direction"],
                "carton_count": int(row["carton_count"]),
                "carrying_mode": row.get("carrying_mode", "single_worker"),
                "notes": row.get("notes", ""),
            })
    return events


class TestEvaluatorLogicRegression:
    """Mathematical verification of evaluator calculations on synthetic test vectors."""

    def test_synthetic_exact_matching_math(self):
        """Verify CLEM calculation when simulated predictions align within 0.1s."""
        gt_events = [
            {"event_id": f"gt_{i}", "timestamp_sec": float(i * 4.0), "direction": "IN", "carton_count": 1}
            for i in range(1, 7)
        ]
        pred_events = [
            {
                "event_id": f"pred_{i}",
                "timestamp_sec": gt["timestamp_sec"] + 0.05,
                "direction": "IN",
                "carton_count": 1,
                "consensus": {"level": "THREE_CAMERA_VERIFIED", "action": "COMMIT"},
                "is_auto_confirmed": True,
            }
            for i, gt in enumerate(gt_events)
        ]

        evaluator = CartonLevelEvaluator(match_window_sec=1.5, clem_target_pct=90.0)
        report: Week5EvaluationReport = evaluator.evaluate(
            ground_truth_events=gt_events,
            predicted_events=pred_events,
        )

        assert report.total_ground_truth_cartons == 6
        assert report.total_predicted_cartons == 6
        assert report.exact_match_events == 6
        assert report.clem_rate_pct == 100.0
        assert report.overcount_events == 0
        assert report.undercount_events == 0
        assert report.missed_events == 0
        assert report.spurious_events == 0
        assert report.net_carton_discrepancy == 0

    def test_overcount_and_undercount_error_metrics(self):
        """Verify accurate diagnosis of over-count, under-count, and CLEM calculation."""
        gt_events = [
            {"event_id": "gt_1", "timestamp_sec": 2.0, "carton_count": 1, "direction": "IN"},
            {"event_id": "gt_2", "timestamp_sec": 5.0, "carton_count": 2, "direction": "IN"},
            {"event_id": "gt_3", "timestamp_sec": 8.0, "carton_count": 1, "direction": "IN"},
            {"event_id": "gt_4", "timestamp_sec": 12.0, "carton_count": 1, "direction": "IN"},
        ]

        pred_events = [
            {"event_id": "p_1", "timestamp_sec": 2.1, "carton_count": 1, "direction": "IN"},
            {"event_id": "p_2", "timestamp_sec": 5.2, "carton_count": 1, "direction": "IN"},  # undercount
            {"event_id": "p_3", "timestamp_sec": 8.0, "carton_count": 1, "direction": "IN"},
            {"event_id": "p_ghost", "timestamp_sec": 15.0, "carton_count": 1, "direction": "IN"},  # spurious
        ]

        evaluator = CartonLevelEvaluator(match_window_sec=1.0)
        report = evaluator.evaluate(gt_events, pred_events)

        assert report.total_ground_truth_events == 4
        assert report.total_ground_truth_cartons == 5
        assert report.total_predicted_cartons == 4
        assert report.exact_match_events == 2
        assert report.undercount_events == 1
        assert report.missed_events == 1
        assert report.spurious_events == 1
        assert report.net_carton_discrepancy == -1

    def test_robustness_under_visual_occlusion_and_shift(self):
        """Consensus logic verification when side camera is occluded."""
        pipeline = ThreeCameraCartonPipeline(
            camera_ids=("cam_top", "cam_side", "cam_flank"),
            fps=30.0,
            conf_threshold=0.20,
            require_calibrated_fusion=False,
            allow_temporal_video_fusion=True,
        )

        cand_top = LocalCameraCrossingCandidate(
            local_event_id="top_ev_01",
            local_track_id=101,
            camera_id="cam_top",
            timestamp_sec=3.0,
            observed_frame_idx=90,
            direction="IN",
            confidence=0.88,
            bbox=(450.0, 200.0, 550.0, 300.0),
            aspect_ratio=1.0,
            crossing_line_px=500.0,
            crossing_point_px=(500.0, 250.0),
        )
        cand_flank = LocalCameraCrossingCandidate(
            local_event_id="flank_ev_01",
            local_track_id=201,
            camera_id="cam_flank",
            timestamp_sec=3.15,
            observed_frame_idx=95,
            direction="IN",
            confidence=0.76,
            bbox=(480.0, 150.0, 580.0, 250.0),
            aspect_ratio=1.0,
            crossing_line_px=500.0,
            crossing_point_px=(500.0, 200.0),
        )

        pipeline._add_crossing_candidate_to_window(cand_top, arrival_time=3.0)
        pipeline._add_crossing_candidate_to_window(cand_flank, arrival_time=3.15)

        flushed = pipeline.flush_pending_windows(current_time_sec=5.0)
        assert len(flushed) == 1
        assert flushed[0]["is_auto_confirmed"] == 1
        assert "cam_top" in flushed[0]["participating_cameras"]
        assert "cam_flank" in flushed[0]["participating_cameras"]
        assert "cam_side" not in flushed[0]["participating_cameras"]
        assert pipeline.confirmed_in == 1

    def test_multi_camera_consensus_breakdown_metrics(self):
        """Consensus policy distinction between unanimous and single-camera observation."""
        pipeline = ThreeCameraCartonPipeline(
            camera_ids=("cam_top", "cam_side", "cam_flank"),
            conf_threshold=0.20,
            require_calibrated_fusion=False,
            allow_temporal_video_fusion=True,
        )

        for cid, tid in [("cam_top", 1), ("cam_side", 2), ("cam_flank", 3)]:
            pipeline._add_crossing_candidate_to_window(
                LocalCameraCrossingCandidate(
                    local_event_id=f"{cid}_w1",
                    local_track_id=tid,
                    camera_id=cid,
                    timestamp_sec=1.0,
                    observed_frame_idx=30,
                    direction="IN",
                    confidence=0.85,
                    bbox=(480.0, 100.0, 520.0, 200.0),
                    aspect_ratio=1.0,
                    crossing_line_px=500.0,
                    crossing_point_px=(500.0, 150.0),
                ),
                arrival_time=1.0,
            )

        pipeline._add_crossing_candidate_to_window(
            LocalCameraCrossingCandidate(
                local_event_id="side_w2",
                local_track_id=99,
                camera_id="cam_side",
                timestamp_sec=6.0,
                observed_frame_idx=180,
                direction="IN",
                confidence=0.45,
                bbox=(480.0, 100.0, 520.0, 200.0),
                aspect_ratio=1.0,
                crossing_line_px=500.0,
                crossing_point_px=(500.0, 150.0),
            ),
            arrival_time=6.0,
        )

        pipeline.flush_pending_windows(current_time_sec=10.0)

        assert pipeline.confirmed_in == 1
        assert pipeline.review_required_in == 1
        assert pipeline.total_in == 2

