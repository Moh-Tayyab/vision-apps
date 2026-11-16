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

    def test_late_arriving_earlier_timestamp_is_same_carton(self):
        """A camera crossing processed later but timestamped earlier is the same carton.

        The multi-camera feed can deliver one camera's crossing after the shared
        window was already committed, even though that crossing happened a
        fraction of a second earlier. It must be absorbed into the committed
        physical carton instead of being counted a second time.
        """
        pipeline = ThreeCameraCartonPipeline(
            camera_ids=("cam_top", "cam_side", "cam_flank"),
            conf_threshold=0.20,
            require_calibrated_fusion=False,
            allow_temporal_video_fusion=True,
        )

        def candidate(cam: str, tid: int, ts: float, name: str):
            return LocalCameraCrossingCandidate(
                local_event_id=name,
                local_track_id=tid,
                camera_id=cam,
                timestamp_sec=ts,
                observed_frame_idx=int(ts * 30),
                direction="IN",
                confidence=0.85,
                bbox=(480.0, 100.0, 520.0, 200.0),
                aspect_ratio=1.0,
                crossing_line_px=500.0,
                crossing_point_px=(500.0, 150.0),
            )

        pipeline._add_crossing_candidate_to_window(
            candidate("cam_side", 2, 11.26, "side_w1"), arrival_time=11.26
        )
        pipeline._add_crossing_candidate_to_window(
            candidate("cam_flank", 3, 11.24, "flank_w1"), arrival_time=11.24
        )
        first = pipeline.flush_pending_windows(current_time_sec=14.0)
        assert len(first) == 1
        assert pipeline.total_in == 1

        pipeline._add_crossing_candidate_to_window(
            candidate("cam_top", 1, 11.21, "top_w1"), arrival_time=11.40
        )
        pipeline.flush_pending_windows(current_time_sec=14.0)

        assert pipeline.total_in == 1
        assert len(pipeline.audit_events) == 1
        assert pipeline.confirmed_in == 1

    def test_distinct_cartons_outside_debounce_are_counted_separately(self):
        """Two genuinely separate crossings must not be absorbed by the debounce."""
        pipeline = ThreeCameraCartonPipeline(
            camera_ids=("cam_top", "cam_side", "cam_flank"),
            conf_threshold=0.20,
            require_calibrated_fusion=False,
            allow_temporal_video_fusion=True,
        )

        def candidate(cam: str, tid: int, ts: float, name: str):
            return LocalCameraCrossingCandidate(
                local_event_id=name,
                local_track_id=tid,
                camera_id=cam,
                timestamp_sec=ts,
                observed_frame_idx=int(ts * 30),
                direction="IN",
                confidence=0.85,
                bbox=(480.0, 100.0, 520.0, 200.0),
                aspect_ratio=1.0,
                crossing_line_px=500.0,
                crossing_point_px=(500.0, 150.0),
            )

        pipeline._add_crossing_candidate_to_window(
            candidate("cam_side", 2, 1.0, "side_a"), arrival_time=1.0
        )
        pipeline._add_crossing_candidate_to_window(
            candidate("cam_flank", 3, 1.1, "flank_a"), arrival_time=1.1
        )
        pipeline.flush_pending_windows(current_time_sec=4.0)
        assert pipeline.total_in == 1

        pipeline._add_crossing_candidate_to_window(
            candidate("cam_side", 4, 3.0, "side_b"), arrival_time=3.0
        )
        pipeline._add_crossing_candidate_to_window(
            candidate("cam_flank", 5, 3.1, "flank_b"), arrival_time=3.1
        )
        pipeline.flush_pending_windows(current_time_sec=6.0)

        assert pipeline.total_in == 2
        assert len(pipeline.audit_events) == 2

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


class TestWorkerFalsePositiveRejection:
    """A carton-only model fires on worker clothing and limbs.

    Those detections must never reach the counting logic, otherwise they are
    stitched to a stale approach record and fabricate a phantom load event.
    """

    @staticmethod
    def _rect_polygon(x1: int, y1: int, x2: int, y2: int) -> List[List[float]]:
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def test_low_confidence_mask_inside_worker_is_false_positive(self):
        from model_pipeline.mask_refinement import CartonMaskRefiner

        refiner = CartonMaskRefiner()
        worker = self._rect_polygon(700, 0, 1200, 600)
        clothing_mask = self._rect_polygon(850, 77, 1101, 378)

        assert refiner.worker_false_positive(
            raw_polygon=clothing_mask,
            image_shape=(720, 1280),
            worker_polygons=[worker],
            confidence=0.15,
        ) is True

    def test_carton_only_partially_occluded_is_not_false_positive(self):
        from model_pipeline.mask_refinement import CartonMaskRefiner

        refiner = CartonMaskRefiner()
        worker = self._rect_polygon(556, 0, 960, 544)
        carton_mask = self._rect_polygon(873, 13, 1034, 214)

        assert refiner.worker_false_positive(
            raw_polygon=carton_mask,
            image_shape=(720, 1280),
            worker_polygons=[worker],
            confidence=0.16,
        ) is False

    def test_high_confidence_overlap_is_not_false_positive(self):
        from model_pipeline.mask_refinement import CartonMaskRefiner

        refiner = CartonMaskRefiner()
        worker = self._rect_polygon(700, 0, 1200, 600)
        mask = self._rect_polygon(850, 77, 1101, 378)

        assert refiner.worker_false_positive(
            raw_polygon=mask,
            image_shape=(720, 1280),
            worker_polygons=[worker],
            confidence=0.80,
        ) is False

    def test_adapter_drops_worker_false_positive_detection(self):
        import numpy as np
        import torch

        from model_pipeline.mask_refinement import CartonMaskRefiner
        from three_camera_system.detector_adapter import CartonDetectorAdapter

        class _FakeBox:
            def __init__(self, xyxy, conf):
                self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
                self.conf = torch.tensor([conf], dtype=torch.float32)
                self.cls = torch.tensor([0], dtype=torch.float32)

        class _FakeMasks:
            def __init__(self, polys):
                self.xy = polys

        class _FakeResult:
            def __init__(self, xyxy, conf, polys):
                self.boxes = [_FakeBox(xyxy, conf)]
                self.masks = _FakeMasks(polys)

        adapter = CartonDetectorAdapter.__new__(CartonDetectorAdapter)
        adapter.refiner = CartonMaskRefiner()
        adapter._names = {0: "carton"}
        adapter._carton_class_ids = [0]

        worker = np.array(self._rect_polygon(700, 0, 1200, 600), dtype=np.float32)
        clothing = np.array(self._rect_polygon(850, 77, 1101, 378), dtype=np.float32)
        result = _FakeResult([850.0, 77.0, 1101.0, 378.0], 0.15, [clothing])

        detections = adapter._parse_results(
            result,
            "cam_flank",
            0,
            18.57,
            1280,
            720,
            worker_polygons=[worker],
        )

        assert detections == []

    def test_adapter_keeps_genuine_carton_detection(self):
        import numpy as np
        import torch

        from model_pipeline.mask_refinement import CartonMaskRefiner
        from three_camera_system.detector_adapter import CartonDetectorAdapter

        class _FakeBox:
            def __init__(self, xyxy, conf):
                self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
                self.conf = torch.tensor([conf], dtype=torch.float32)
                self.cls = torch.tensor([0], dtype=torch.float32)

        class _FakeMasks:
            def __init__(self, polys):
                self.xy = polys

        class _FakeResult:
            def __init__(self, xyxy, conf, polys):
                self.boxes = [_FakeBox(xyxy, conf)]
                self.masks = _FakeMasks(polys)

        adapter = CartonDetectorAdapter.__new__(CartonDetectorAdapter)
        adapter.refiner = CartonMaskRefiner()
        adapter._names = {0: "carton"}
        adapter._carton_class_ids = [0]

        worker = np.array(self._rect_polygon(556, 0, 960, 544), dtype=np.float32)
        carton = np.array(self._rect_polygon(873, 13, 1034, 214), dtype=np.float32)
        result = _FakeResult([873.0, 13.0, 1034.0, 214.0], 0.16, [carton])

        detections = adapter._parse_results(
            result,
            "cam_flank",
            0,
            26.57,
            1280,
            720,
            worker_polygons=[worker],
        )

        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.16)



class TestContentCropCoordinateMapping:
    """Normalized videos pad a narrow camera into a wide canvas.

    The padded frame starves the model input, so the visible content is
    cropped before inference. Every coordinate must be mapped back to the
    original full-frame space, otherwise tripwire geometry is invalidated.
    """

    @staticmethod
    def _pillarboxed_frame() -> "np.ndarray":
        import numpy as np

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, 438:842] = 80
        return frame

    def test_crop_is_disabled_by_default(self):
        from three_camera_system.detector_adapter import CartonDetectorAdapter

        adapter = CartonDetectorAdapter.__new__(CartonDetectorAdapter)
        adapter.enable_content_crop = False
        adapter.content_crop_margin_px = 4
        adapter.content_crop_min_coverage = 0.92

        x0, x1 = adapter._content_crop(self._pillarboxed_frame())

        assert (x0, x1) == (0, 1280)

    def test_crop_selects_visible_content_strip(self):
        from three_camera_system.detector_adapter import CartonDetectorAdapter

        adapter = CartonDetectorAdapter.__new__(CartonDetectorAdapter)
        adapter.enable_content_crop = True
        adapter.content_crop_margin_px = 4
        adapter.content_crop_min_coverage = 0.92

        x0, x1 = adapter._content_crop(self._pillarboxed_frame())

        assert x0 == 434
        assert x1 == 846
        assert (x1 - x0) < 1280

    def test_parse_results_maps_boxes_back_to_full_frame(self):
        import numpy as np
        import torch

        from model_pipeline.mask_refinement import CartonMaskRefiner
        from three_camera_system.detector_adapter import CartonDetectorAdapter

        class _FakeBox:
            def __init__(self, xyxy, conf):
                self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
                self.conf = torch.tensor([conf], dtype=torch.float32)
                self.cls = torch.tensor([0], dtype=torch.float32)

        class _FakeMasks:
            def __init__(self, polys):
                self.xy = polys

        class _FakeResult:
            def __init__(self, xyxy, conf, polys):
                self.boxes = [_FakeBox(xyxy, conf)]
                self.masks = _FakeMasks(polys)

        adapter = CartonDetectorAdapter.__new__(CartonDetectorAdapter)
        adapter.refiner = CartonMaskRefiner()
        adapter._names = {0: "carton"}
        adapter._carton_class_ids = [0]

        # Box detected inside the cropped strip (crop origin x=434).
        carton = np.array(
            [[440, 10], [520, 10], [520, 90], [440, 90]], dtype=np.float32
        )
        result = _FakeResult([440.0, 10.0, 520.0, 90.0], 0.72, [carton])

        detections = adapter._parse_results(
            result,
            "cam_top",
            0,
            1.0,
            1280,
            720,
            x_offset=434,
        )
        unshifted = adapter._parse_results(
            result,
            "cam_top",
            0,
            1.0,
            1280,
            720,
            x_offset=0,
        )

        assert len(detections) == 1
        assert len(unshifted) == 1
        assert detections[0].bbox[0] == pytest.approx(unshifted[0].bbox[0] + 434.0)
        assert detections[0].bbox[2] == pytest.approx(unshifted[0].bbox[2] + 434.0)
