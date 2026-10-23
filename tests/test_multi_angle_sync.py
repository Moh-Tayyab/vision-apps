import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "truck_loading_detection"))

from multi_angle_sync import estimate_offset, offset_curve, resolve_offsets
from multi_angle_fusion_engine import (
    CameraAngleConfig,
    MultiAngleFusionEngine,
)


def _events(pairs):
    """pairs: list of (direction, timestamp) -> engine-style event dicts."""
    out = []
    for i, (d, t) in enumerate(pairs):
        out.append({
            "camera_id": "cam",
            "angle_type": "side_profile",
            "track_id": i,
            "class_name": "person",
            "confidence": 0.9,
            "direction": d,
            "carton_qty": 1,
            "holding": True,
            "frame_idx": i,
            "timestamp_sec": t,
            "bbox": (0, 0, 10, 10),
        })
    return out


# ---------------------------------------------------------------- sync module

def test_estimate_offset_recovers_known_lag():
    ref = _events([("IN", 1.0), ("OUT", 5.0), ("IN", 9.0), ("OUT", 13.0)])
    # other camera started 3.5s LATER -> its local times are 3.5s earlier
    other = _events([("IN", 1.0 - 3.5), ("OUT", 5.0 - 3.5), ("IN", 9.0 - 3.5)])
    est = estimate_offset(ref, other, max_offset_sec=15.0, tol_sec=0.5)
    assert abs(est["offset_sec"] - 3.5) < 0.1
    assert est["matched_events"] == 3
    assert est["confidence"] > 0.5


def test_estimate_offset_is_zero_for_identical_timelines():
    ref = _events([("IN", 2.0), ("OUT", 6.0)])
    est = estimate_offset(ref, ref, max_offset_sec=10.0, tol_sec=0.3)
    assert abs(est["offset_sec"]) < 0.05
    assert est["matched_events"] == 2


def test_estimate_offset_respects_direction():
    """Reversed directions must not be matched at the same timestamp."""
    ref = _events([("IN", 2.0), ("OUT", 6.0)])
    other = _events([("OUT", 2.0), ("IN", 6.0)])
    curve = offset_curve(ref, other, max_offset_sec=0.5, tol_sec=0.3)
    assert all(matched == 0 for _off, matched, _res in curve)


def test_resolve_offsets_uses_reference_and_keeps_errors_local():
    per_cam = {
        "cam_a": _events([("IN", 1.0), ("OUT", 4.0), ("IN", 7.0)]),
        "cam_b": _events([("IN", 1.0 - 2.0), ("OUT", 4.0 - 2.0), ("IN", 7.0 - 2.0)]),
        "cam_c": _events([("IN", 1.0 + 5.0), ("OUT", 4.0 + 5.0), ("IN", 7.0 + 5.0)]),
    }
    res = resolve_offsets(per_cam, reference_camera_id="cam_a", tol_sec=0.4)
    assert res["reference"] == "cam_a"
    assert res["offsets"]["cam_a"] == 0.0
    assert abs(res["offsets"]["cam_b"] - 2.0) < 0.1
    assert abs(res["offsets"]["cam_c"] - (-5.0)) < 0.1


def test_resolve_offsets_picks_reference_with_most_events():
    per_cam = {
        "few": _events([("IN", 1.0)]),
        "many": _events([("IN", 1.0), ("OUT", 2.0), ("IN", 3.0), ("OUT", 4.0)]),
    }
    res = resolve_offsets(per_cam)
    assert res["reference"] == "many"


# ------------------------------------------------------- 3-camera consensus

def test_three_camera_consensus_marks_three_camera_verified():
    engine = MultiAngleFusionEngine(time_window_sec=0.5)
    engine.add_camera(CameraAngleConfig("cam_top", "top_overhead", "horizontal", 400, "up_to_down"))
    engine.add_camera(CameraAngleConfig("cam_side", "side_profile", "vertical", 500, "left_to_right"))
    engine.add_camera(CameraAngleConfig("cam_flank", "side_profile", "vertical", 500,
                                        "left_to_right", role="corroborator"))

    def cross_horizontal(cam, y, qty):
        """Top camera counts on a horizontal line, so the worker must move in Y."""
        engine.update_camera_frame(cam, [(1, (200.0, y, 260.0, y + 60.0), "person", 0.9, qty)], 1, 1.0)
        engine.update_camera_frame(cam, [(1, (200.0, y + 100.0, 260.0, y + 160.0), "person", 0.9, qty)], 2, 1.1)

    def cross_vertical(cam, x, qty):
        """Side/flank cameras count on a vertical line, so the worker moves in X."""
        engine.update_camera_frame(cam, [(1, (x, 200.0, x + 60.0, 400.0), "person", 0.9, qty)], 1, 1.0)
        engine.update_camera_frame(cam, [(1, (x + 100.0, 200.0, x + 160.0, 400.0), "person", 0.9, qty)], 2, 1.1)

    cross_horizontal("cam_top", 300.0, 2)
    cross_vertical("cam_side", 400.0, 2)
    cross_vertical("cam_flank", 400.0, 1)

    fused = engine.process_time_window_consensus(current_time_sec=1.3)
    assert len(fused) == 1
    f = fused[0]
    assert f["consensus_status"] == "THREE_CAMERA_VERIFIED"
    assert f["top_camera_width"] == 2
    assert f["side_camera_height"] == 2
    assert f["fused_cartons"] == 4
    assert f["corroborating_cameras"] == ["cam_flank"]


def test_two_camera_consensus_stays_dual_camera_verified():
    engine = MultiAngleFusionEngine(time_window_sec=0.5)
    engine.add_camera(CameraAngleConfig("cam_top", "top_overhead", "horizontal", 400, "up_to_down"))
    engine.add_camera(CameraAngleConfig("cam_side", "side_profile", "vertical", 500, "left_to_right"))

    engine.update_camera_frame("cam_top", [(1, (200.0, 300.0, 260.0, 360.0), "person", 0.9, 2)], 1, 1.0)
    engine.update_camera_frame("cam_top", [(1, (200.0, 450.0, 260.0, 510.0), "person", 0.9, 2)], 2, 1.1)
    engine.update_camera_frame("cam_side", [(2, (400.0, 200.0, 460.0, 400.0), "person", 0.9, 2)], 1, 1.0)
    engine.update_camera_frame("cam_side", [(2, (540.0, 200.0, 600.0, 400.0), "person", 0.9, 2)], 2, 1.12)

    fused = engine.process_time_window_consensus(current_time_sec=1.3)
    assert len(fused) == 1
    assert fused[0]["consensus_status"] == "DUAL_CAMERA_VERIFIED"
    assert fused[0]["fused_cartons"] == 4


def test_empty_handed_crossing_counts_zero_cartons():
    """A worker crossing with no carton must not be counted as one carton."""
    engine = MultiAngleFusionEngine(time_window_sec=0.5)
    engine.add_camera(CameraAngleConfig("cam_side", "side_profile", "vertical", 500, "left_to_right"))

    engine.update_camera_frame("cam_side", [(1, (400.0, 200.0, 460.0, 400.0), "person", 0.9, 0)], 1, 1.0)
    engine.update_camera_frame("cam_side", [(1, (540.0, 200.0, 600.0, 400.0), "person", 0.9, 0)], 2, 1.1)

    # No width camera is present, so the height-only fallback only fires once
    # the fusion window has expired.
    fused = engine.process_time_window_consensus(current_time_sec=1.8)
    assert len(fused) == 1
    assert fused[0]["fused_cartons"] == 0
    assert engine.total_loaded_cartons == 0
    assert engine.total_returned_cartons == 0


def test_corroborator_alone_never_produces_a_count():
    engine = MultiAngleFusionEngine(time_window_sec=0.5)
    engine.add_camera(CameraAngleConfig("cam_flank", "side_profile", "vertical", 500,
                                        "left_to_right", role="corroborator"))
    engine.update_camera_frame("cam_flank", [(1, (400.0, 200.0, 460.0, 400.0), "person", 0.9, 2)], 1, 1.0)
    engine.update_camera_frame("cam_flank", [(1, (540.0, 200.0, 600.0, 400.0), "person", 0.9, 2)], 2, 1.1)

    fused = engine.process_time_window_consensus(current_time_sec=1.3)
    assert fused == []
    assert engine.total_loaded_cartons == 0
    assert engine.total_worker_trips == 0


# ------------------------------------------------------- width sampler role

def test_width_sampler_supplies_w_for_a_height_event():
    """
    A camera that cannot produce tripwire crossings (handheld overhead) can
    still report how WIDE a load is; the height camera's event then fuses
    W (sampled) x H (measured) instead of falling back to H alone.
    """
    engine = MultiAngleFusionEngine(time_window_sec=0.8)
    engine.add_camera(CameraAngleConfig("cam_top", "top_overhead", "horizontal", 400,
                                        "up_to_down", role="width_sampler"))
    engine.add_camera(CameraAngleConfig("cam_side", "side_profile", "vertical", 500, "left_to_right"))

    # Overhead reports a load 2 cartons wide at t = 1.0
    engine.update_camera_frame("cam_top", [(1, (10.0, 10.0, 60.0, 60.0), "person", 0.9, 2)], 1, 1.0)

    # Side camera sees the same worker cross with a stack of 2
    engine.update_camera_frame("cam_side", [(2, (400.0, 200.0, 460.0, 400.0), "person", 0.9, 2)], 1, 1.0)
    engine.update_camera_frame("cam_side", [(2, (540.0, 200.0, 600.0, 400.0), "person", 0.9, 2)], 2, 1.1)

    fused = engine.process_time_window_consensus(current_time_sec=2.0)
    assert len(fused) == 1
    f = fused[0]
    assert f["consensus_status"] == "WIDTH_SAMPLED_VERIFIED"
    assert f["top_camera_width"] == 2
    assert f["side_camera_height"] == 2
    assert f["fused_cartons"] == 4
    assert f["width_sampler"] == "cam_top"
    assert engine.total_loaded_cartons == 4


def test_width_sampler_ignores_empty_handed_detections():
    engine = MultiAngleFusionEngine(time_window_sec=0.8)
    engine.add_camera(CameraAngleConfig("cam_top", "top_overhead", "horizontal", 400,
                                        "up_to_down", role="width_sampler"))
    engine.add_camera(CameraAngleConfig("cam_side", "side_profile", "vertical", 500, "left_to_right"))

    # Overhead: worker is empty handed -> no width reading
    engine.update_camera_frame("cam_top", [(1, (10.0, 10.0, 60.0, 60.0), "person", 0.9, 0)], 1, 1.0)
    engine.update_camera_frame("cam_side", [(2, (400.0, 200.0, 460.0, 400.0), "person", 0.9, 2)], 1, 1.0)
    engine.update_camera_frame("cam_side", [(2, (540.0, 200.0, 600.0, 400.0), "person", 0.9, 2)], 2, 1.1)

    fused = engine.process_time_window_consensus(current_time_sec=2.0)
    assert len(fused) == 1
    assert fused[0]["consensus_status"] == "HEIGHT_ONLY_FALLBACK"
    assert fused[0]["top_camera_width"] == 1
    assert fused[0]["fused_cartons"] == 2
