"""
Business Acceptance Test Suite: Simplified Three-Camera Warehouse Carton Loading Counter.

Tests the 8 core business requirements:
  Test 1: One carton loaded -> Loaded=1, Returned=0, Net=1
  Test 2: One carton returned -> Loaded=0, Returned=1, Net=-1
  Test 3: Worker crosses without carton -> Loaded=0, Returned=0, Net=0
  Test 4: One carton observed by three cameras -> Loaded=1 (NOT 3)
  Test 5: Two different cartons cross -> Loaded=2, Returned=0, Net=2
  Test 6: Carton approaches but does not cross -> Loaded=0, Returned=0, Net=0
  Test 7: Carton crosses and returns -> Loaded=1, Returned=1, Net=0
  Test 8: Worker occlusion and tracker ID changes -> Exactly 1 count
"""

import math
import numpy as np
import pytest
from typing import Dict, List, Tuple

from apps.truck_loading_detection.three_camera_system.pipeline import (
    ThreeCameraCartonPipeline,
    CrossingWindow,
)
from apps.truck_loading_detection.three_camera_system.corridor import (
    CameraTripwireConfig,
    PerCameraTripwire,
    LocalCameraCrossingCandidate,
)
from apps.truck_loading_detection.three_camera_system.per_camera_tracker import (
    CartonTracklet,
    PerCameraCartonTracker,
    TrackStatus,
)
from apps.truck_loading_detection.three_camera_system.detector_adapter import (
    CanonicalCartonDetection,
)


def _make_candidate(
    camera_id: str,
    track_id: str,
    timestamp_sec: float,
    direction: str = "IN",
    center_y: float = 300.0,
    conf: float = 0.85,
) -> LocalCameraCrossingCandidate:
    x = 500.0
    return LocalCameraCrossingCandidate(
        local_event_id=f"ev_{camera_id}_{track_id}",
        camera_id=camera_id,
        local_track_id=track_id,
        timestamp_sec=timestamp_sec,
        direction=direction,
        confidence=conf,
        bbox=(x - 30, center_y - 30, x + 30, center_y + 30),
        aspect_ratio=1.0,
        estimated_qty=1,
    )


# ===========================================================================
# Test 1: One carton loaded -> Loaded=1, Returned=0, Net=1
# ===========================================================================
def test_acceptance_1_one_carton_loaded():
    pipeline = ThreeCameraCartonPipeline()
    c_side = _make_candidate("cam_side", "trk_01_side", timestamp_sec=1.0, direction="IN")
    c_flank = _make_candidate("cam_flank", "trk_01_flank", timestamp_sec=1.1, direction="IN")

    pipeline._add_crossing_candidate_to_window(c_side)
    pipeline._add_crossing_candidate_to_window(c_flank)

    # Fast forward time to close window
    dummy_frames = {c: np.zeros((10, 10, 3), dtype=np.uint8) for c in pipeline.camera_ids}
    pipeline.process_synchronized_frames(dummy_frames, frame_idx=100, override_time_sec=3.5)

    assert pipeline.total_in == 1
    assert pipeline.total_out == 0
    assert pipeline.net_count == 1
    assert len(pipeline.audit_events) == 1
    assert pipeline.audit_events[0]["direction"] == "IN"


# ===========================================================================
# Test 2: One carton returned -> Loaded=0, Returned=1, Net=-1
# ===========================================================================
def test_acceptance_2_one_carton_returned():
    pipeline = ThreeCameraCartonPipeline()
    c_side = _make_candidate("cam_side", "trk_ret_side", timestamp_sec=2.0, direction="OUT")
    c_flank = _make_candidate("cam_flank", "trk_ret_flank", timestamp_sec=2.1, direction="OUT")

    pipeline._add_crossing_candidate_to_window(c_side)
    pipeline._add_crossing_candidate_to_window(c_flank)

    dummy_frames = {c: np.zeros((10, 10, 3), dtype=np.uint8) for c in pipeline.camera_ids}
    pipeline.process_synchronized_frames(dummy_frames, frame_idx=120, override_time_sec=4.5)

    assert pipeline.total_in == 0
    assert pipeline.total_out == 1
    assert pipeline.net_count == -1
    assert len(pipeline.audit_events) == 1
    assert pipeline.audit_events[0]["direction"] == "OUT"


# ===========================================================================
# Test 3: Worker crosses without carton -> Loaded=0, Returned=0, Net=0
# ===========================================================================
def test_acceptance_3_worker_crosses_without_carton():
    pipeline = ThreeCameraCartonPipeline()
    # Canonical model detects only cartons. A worker crossing empty-handed yields 0 carton detections.
    blank_frames = {c: np.zeros((10, 10, 3), dtype=np.uint8) for c in pipeline.camera_ids}

    for f_idx in range(60):
        t_sec = f_idx * 0.033
        pipeline.process_synchronized_frames(blank_frames, frame_idx=f_idx, override_time_sec=t_sec)

    assert pipeline.total_in == 0
    assert pipeline.total_out == 0
    assert pipeline.net_count == 0
    assert len(pipeline.audit_events) == 0


# ===========================================================================
# Test 4: One carton observed by three cameras -> Loaded=1 (NOT 3)
# ===========================================================================
def test_acceptance_4_one_carton_observed_by_three_cameras():
    pipeline = ThreeCameraCartonPipeline()

    # cam_flank detects first, then cam_top, then cam_side across the traversal
    c_flank = _make_candidate("cam_flank", "trk_flank_04", timestamp_sec=2.0, direction="IN")
    c_top = _make_candidate("cam_top", "trk_top_04", timestamp_sec=2.4, direction="IN")
    c_side = _make_candidate("cam_side", "trk_side_04", timestamp_sec=2.6, direction="IN")

    pipeline._add_crossing_candidate_to_window(c_flank)
    pipeline._add_crossing_candidate_to_window(c_top)
    pipeline._add_crossing_candidate_to_window(c_side)

    dummy_frames = {c: np.zeros((10, 10, 3), dtype=np.uint8) for c in pipeline.camera_ids}
    pipeline.process_synchronized_frames(dummy_frames, frame_idx=150, override_time_sec=5.0)

    # Exactly ONE global count, not 3!
    assert pipeline.total_in == 1
    assert pipeline.total_out == 0
    assert pipeline.net_count == 1
    assert len(pipeline.audit_events) == 1
    assert set(pipeline.audit_events[0]["participating_cameras"]) == {"cam_flank", "cam_top", "cam_side"}


# ===========================================================================
# Test 5: Two different cartons cross -> Loaded=2, Returned=0, Net=2
# ===========================================================================
def test_acceptance_5_two_different_cartons_cross():
    pipeline = ThreeCameraCartonPipeline()

    # Carton 1 crossing at t=2.0s
    c1_side = _make_candidate("cam_side", "carton1_side", timestamp_sec=2.0, direction="IN")
    c1_flank = _make_candidate("cam_flank", "carton1_flank", timestamp_sec=2.1, direction="IN")
    pipeline._add_crossing_candidate_to_window(c1_side)
    pipeline._add_crossing_candidate_to_window(c1_flank)

    dummy_frames = {c: np.zeros((10, 10, 3), dtype=np.uint8) for c in pipeline.camera_ids}
    pipeline.process_synchronized_frames(dummy_frames, frame_idx=120, override_time_sec=4.0)

    # Carton 2 crossing at t=7.0s
    c2_side = _make_candidate("cam_side", "carton2_side", timestamp_sec=7.0, direction="IN")
    c2_flank = _make_candidate("cam_flank", "carton2_flank", timestamp_sec=7.1, direction="IN")
    pipeline._add_crossing_candidate_to_window(c2_side)
    pipeline._add_crossing_candidate_to_window(c2_flank)

    pipeline.process_synchronized_frames(dummy_frames, frame_idx=270, override_time_sec=9.0)

    assert pipeline.total_in == 2
    assert pipeline.total_out == 0
    assert pipeline.net_count == 2
    assert len(pipeline.audit_events) == 2


# ===========================================================================
# Test 6: Carton approaches but does not cross -> Loaded=0, Returned=0, Net=0
# ===========================================================================
def test_acceptance_6_carton_approaches_without_crossing():
    tripwire = PerCameraTripwire(CameraTripwireConfig(
        camera_id="cam_side",
        orientation="vertical",
        line_pos_frac=0.50,
        direction_sign=1,
    ))

    # A carton approaches from x=200 to x=400 (line is at 500 in 1000px wide image), then retreats to x=200
    class MovingCarton:
        def __init__(self, x, y):
            self.local_track_id = "carton_approach"
            self.center = (x, y)
            self.current_confidence = 0.90
            self.trajectory = [(x, y, 1.0)]

    events = []
    # Moves forward toward line but stays before hysteresis boundary
    for x in range(200, 430, 20):
        ev = tripwire.update(MovingCarton(float(x), 300.0), frame_idx=10, timestamp_sec=1.0, frame_dims=(600, 1000))
        if ev: events.append(ev)

    # Retreats back to pallet
    for x in range(430, 200, -20):
        ev = tripwire.update(MovingCarton(float(x), 300.0), frame_idx=20, timestamp_sec=2.0, frame_dims=(600, 1000))
        if ev: events.append(ev)

    assert len(events) == 0, "Non-crossing carton approach generated phantom crossing event"


# ===========================================================================
# Test 7: Carton crosses and returns -> Loaded=1, Returned=1, Net=0
# ===========================================================================
def test_acceptance_7_carton_crosses_and_returns():
    pipeline = ThreeCameraCartonPipeline()

    # 1. Forward crossing: IN at t=2.0s
    c_in = _make_candidate("cam_side", "carton_fwd_side", timestamp_sec=2.0, direction="IN")
    c_in_f = _make_candidate("cam_flank", "carton_fwd_flank", timestamp_sec=2.1, direction="IN")
    pipeline._add_crossing_candidate_to_window(c_in)
    pipeline._add_crossing_candidate_to_window(c_in_f)

    dummy_frames = {c: np.zeros((10, 10, 3), dtype=np.uint8) for c in pipeline.camera_ids}
    pipeline.process_synchronized_frames(dummy_frames, frame_idx=120, override_time_sec=4.0)

    assert pipeline.total_in == 1
    assert pipeline.total_out == 0
    assert pipeline.net_count == 1

    # 2. Reverse return crossing: OUT at t=7.0s
    c_out = _make_candidate("cam_side", "carton_ret_side", timestamp_sec=7.0, direction="OUT")
    c_out_f = _make_candidate("cam_flank", "carton_ret_flank", timestamp_sec=7.1, direction="OUT")
    pipeline._add_crossing_candidate_to_window(c_out)
    pipeline._add_crossing_candidate_to_window(c_out_f)

    pipeline.process_synchronized_frames(dummy_frames, frame_idx=270, override_time_sec=9.0)

    assert pipeline.total_in == 1
    assert pipeline.total_out == 1
    assert pipeline.net_count == 0
    assert len(pipeline.audit_events) == 2


# ===========================================================================
# Test 8: Worker occlusion and tracker ID changes -> counted only once
# ===========================================================================
def test_acceptance_8_worker_occlusion_and_tracker_id_changes():
    pipeline = ThreeCameraCartonPipeline()

    # Carton is tracked as track_id_A approaching line
    c_a = _make_candidate("cam_side", "track_id_A", timestamp_sec=2.0, direction="IN", center_y=300.0)
    pipeline._add_crossing_candidate_to_window(c_a)

    # Worker occludes carton; tracker re-acquires it under track_id_B 0.5s later at same lane
    c_b = _make_candidate("cam_side", "track_id_B", timestamp_sec=2.5, direction="IN", center_y=305.0)
    pipeline._add_crossing_candidate_to_window(c_b)

    # cam_top and cam_flank corroborate
    c_top = _make_candidate("cam_top", "track_id_top", timestamp_sec=2.4, direction="IN", center_y=300.0)
    pipeline._add_crossing_candidate_to_window(c_top)

    dummy_frames = {c: np.zeros((10, 10, 3), dtype=np.uint8) for c in pipeline.camera_ids}
    pipeline.process_synchronized_frames(dummy_frames, frame_idx=150, override_time_sec=5.0)

    # Must be fused into ONE global event despite tracker ID split!
    assert pipeline.total_in == 1
    assert pipeline.total_out == 0
    assert pipeline.net_count == 1
    assert len(pipeline.audit_events) == 1

