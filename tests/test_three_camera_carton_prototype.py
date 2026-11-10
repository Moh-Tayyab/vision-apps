"""
Comprehensive Test Suite for Three-Camera Carton-Centric Correlation Prototype.

Covers:
- Phase 1: CanonicalCartonDetection and class filtering
- Phase 2: Per-camera tracklet isolation and local ID scoping
- Phase 3: Timestamp alignment, start offsets, and sync jitter rejection
- Phase 4: Geometric calibration validation and missing-calibration fallback
- Phase 5: Cross-camera association, 1-to-1 matching, and duplicate prevention
- Phase 6: Deterministic crossing state machine, boundary jitter, and reversal
- Phase 7: Three-camera, two-camera, and single-camera consensus policies
- Phase 8: Stack disambiguation states (resolved, partial, unresolved, ambiguous)
- Exactly-once counting guarantee under simulated multi-frame crossings
"""

import pytest
import numpy as np
from typing import Tuple, Dict, List, Optional

from apps.truck_loading_detection.three_camera_system.detector_adapter import (
    CanonicalCartonDetection,
)
from apps.truck_loading_detection.three_camera_system.per_camera_tracker import (
    CartonTracklet,
    PerCameraCartonTracker,
    TrackStatus,
)
from apps.truck_loading_detection.three_camera_system.timeline_aligner import (
    CameraSyncConfig,
    SynchronizedTimestamp,
    TimelineAligner,
)
from apps.truck_loading_detection.three_camera_system.geometric_adapter import (
    CameraCalibrationProfile,
    GeometricCalibrationManager,
)
from apps.truck_loading_detection.three_camera_system.cross_camera_associator import (
    AssociatedCartonCluster,
    CrossCameraAssociator,
)
from apps.truck_loading_detection.three_camera_system.crossing_state_machine import (
    CartonCrossingStateMachine,
    CrossingState,
)
from apps.truck_loading_detection.three_camera_system.consensus_policy import (
    ConsensusLevel,
    DecisionAction,
    ThreeCameraConsensusPolicy,
)
from apps.truck_loading_detection.three_camera_system.stack_disambiguator import (
    StackDisambiguator,
    StackResolutionState,
)


# ===========================================================================
# 1. Detector Adapter Tests
# ===========================================================================
def test_canonical_carton_detection_creation():
    det = CanonicalCartonDetection(
        camera_id="cam_top",
        frame_idx=10,
        timestamp_sec=0.333,
        bbox=(100.0, 150.0, 200.0, 250.0),
        confidence=0.88,
        class_name="carton",
    )
    assert det.camera_id == "cam_top"
    assert det.center == (150.0, 200.0)
    assert det.width == 100.0
    assert det.height == 100.0


def test_canonical_detection_rejects_non_carton():
    with pytest.raises(ValueError, match="only accepts 'carton'"):
        CanonicalCartonDetection(
            camera_id="cam_top",
            frame_idx=1,
            timestamp_sec=0.0,
            bbox=(10.0, 10.0, 50.0, 50.0),
            confidence=0.9,
            class_name="person",
        )


# ===========================================================================
# 2. Per-Camera Tracklet Tests
# ===========================================================================
def test_per_camera_tracklet_isolation():
    tracker_top = PerCameraCartonTracker(camera_id="cam_top")
    tracker_side = PerCameraCartonTracker(camera_id="cam_side")

    det1 = CanonicalCartonDetection(
        camera_id="cam_top", frame_idx=1, timestamp_sec=0.033, bbox=(100, 100, 200, 200), confidence=0.8
    )
    det2 = CanonicalCartonDetection(
        camera_id="cam_side", frame_idx=1, timestamp_sec=0.033, bbox=(100, 100, 200, 200), confidence=0.8
    )

    trks_top = tracker_top.update([det1], frame_idx=1, timestamp_sec=0.033)
    trks_side = tracker_side.update([det2], frame_idx=1, timestamp_sec=0.033)

    assert len(trks_top) == 1
    assert len(trks_side) == 1
    # Local track IDs must be prefixed with camera ID and isolated
    assert trks_top[0].local_track_id == "cam_top_t0001"
    assert trks_side[0].local_track_id == "cam_side_t0001"
    assert trks_top[0].local_track_id != trks_side[0].local_track_id


def test_tracklet_coasting_and_termination_on_missing_frames():
    tracker = PerCameraCartonTracker(camera_id="cam_flank", max_invisible_frames=3)
    det = CanonicalCartonDetection(
        camera_id="cam_flank", frame_idx=1, timestamp_sec=0.0, bbox=(50, 50, 100, 100), confidence=0.9
    )
    tracker.update([det], frame_idx=1, timestamp_sec=0.0)

    # Frame 2: Missing detection -> should coast as OCCLUDED
    active = tracker.update([], frame_idx=2, timestamp_sec=0.033)
    assert len(active) == 1
    assert active[0].status == TrackStatus.OCCLUDED

    # Frames 3, 4, 5: Still missing -> reaches termination
    tracker.update([], frame_idx=3, timestamp_sec=0.066)
    tracker.update([], frame_idx=4, timestamp_sec=0.099)
    active_after = tracker.update([], frame_idx=5, timestamp_sec=0.132)
    assert len(active_after) == 0  # Terminated and removed from active


# ===========================================================================
# 3. Timestamp Alignment Tests
# ===========================================================================
def test_timeline_alignment_with_start_offset():
    configs = {
        "cam_top": CameraSyncConfig(camera_id="cam_top", fps=30.0, start_offset_sec=0.0),
        "cam_side": CameraSyncConfig(camera_id="cam_side", fps=30.0, start_offset_sec=0.50),  # Started 0.5s later
    }
    aligner = TimelineAligner(configs)

    ts_top = aligner.compute_timestamp("cam_top", frame_idx=30)
    ts_side = aligner.compute_timestamp("cam_side", frame_idx=30)

    assert ts_top.local_time_sec == 1.0
    assert ts_top.global_time_sec == 1.0

    assert ts_side.local_time_sec == 1.0
    assert ts_side.global_time_sec == 1.50  # 1.0 + 0.5s offset


def test_timeline_rejects_excessive_sync_jitter():
    configs = {
        "cam_top": CameraSyncConfig(camera_id="cam_top", fps=30.0),
        "cam_side": CameraSyncConfig(camera_id="cam_side", fps=30.0),
    }
    aligner = TimelineAligner(configs, max_inter_camera_delta_sec=0.05)

    ts1 = SynchronizedTimestamp("cam_top", 10, 0.33, 0.33, True)
    ts2 = SynchronizedTimestamp("cam_side", 15, 0.50, 0.50, True)

    aligned, skew, reasons = aligner.validate_multi_view_alignment({"cam_top": ts1, "cam_side": ts2})
    assert not aligned
    assert skew == pytest.approx(0.17, abs=1e-3)
    assert len(reasons) > 0


# ===========================================================================
# 4. Geometric Calibration Tests
# ===========================================================================
def test_geometric_adapter_missing_calibration_safety():
    manager = GeometricCalibrationManager()
    assert not manager.is_fully_calibrated(["cam_top", "cam_side", "cam_flank"])

    # Must return None safely rather than crash or fabricate coordinates
    proj = manager.project("cam_side", (500.0, 300.0))
    assert proj is None


def test_geometric_adapter_valid_homography_projection():
    # Identity homography test
    H = np.eye(3, dtype=np.float64)
    prof = CameraCalibrationProfile(camera_id="cam_top", is_calibrated=True, homography_matrix=H)
    manager = GeometricCalibrationManager()
    manager.register_profile(prof)

    res = manager.project("cam_top", (250.0, 180.0))
    assert res is not None
    assert res == (250.0, 180.0)


# ===========================================================================
# 5. Cross-Camera Association Tests
# ===========================================================================
def test_unmeasured_carton_centroids_are_not_cross_camera_associated():
    associator = CrossCameraAssociator(time_window_sec=0.20)

    # 1 tracklet in cam_top, 2 tracklets in cam_side
    trk_top = CartonTracklet(
        camera_id="cam_top",
        local_track_id="cam_top_t0001",
        first_seen_frame=1,
        first_seen_timestamp=1.0,
        last_seen_frame=1,
        last_seen_timestamp=1.0,
        current_bbox=(100, 100, 150, 150),
        current_confidence=0.9,
        status=TrackStatus.CONFIRMED,
    )
    trk_side1 = CartonTracklet(
        camera_id="cam_side",
        local_track_id="cam_side_t0001",
        first_seen_frame=1,
        first_seen_timestamp=1.02,
        last_seen_frame=1,
        last_seen_timestamp=1.02,
        current_bbox=(105, 105, 155, 155),
        current_confidence=0.85,
        status=TrackStatus.CONFIRMED,
    )
    trk_side2 = CartonTracklet(
        camera_id="cam_side",
        local_track_id="cam_side_t0002",
        first_seen_frame=1,
        first_seen_timestamp=1.04,
        last_seen_frame=1,
        last_seen_timestamp=1.04,
        current_bbox=(300, 300, 350, 350),
        current_confidence=0.80,
        status=TrackStatus.CONFIRMED,
    )

    clusters = associator.associate_frame_tracklets(
        camera_tracklets={"cam_top": [trk_top], "cam_side": [trk_side1, trk_side2]},
        current_global_time_sec=1.0,
    )

    # Tracklet centroids are not valid points on the loading-gate plane.
    assert len(clusters) == 3
    assert all(len(cluster.observations) == 1 for cluster in clusters)


# ===========================================================================
# 6. Crossing State Machine & Exactly-Once Counting Tests
# ===========================================================================
def test_crossing_state_machine_exactly_once_guarantee():
    sm = CartonCrossingStateMachine(line_position_px=500.0, hysteresis_margin_px=20.0, required_direction="IN")

    # Step 1: Observed before line (X = 400)
    ev1 = sm.update_carton_position("carton_A", coord=400.0, displacement=10.0, timestamp_sec=1.0, contributing_cameras=["cam_side"])
    assert ev1 is None

    # Step 2: Moves into hysteresis zone (X = 510) -> no commit yet
    ev2 = sm.update_carton_position("carton_A", coord=510.0, displacement=110.0, timestamp_sec=1.1, contributing_cameras=["cam_side"])
    assert ev2 is None

    # Step 3: Crosses beyond line + hysteresis (X = 530) -> COMMITTED!
    ev3 = sm.update_carton_position("carton_A", coord=530.0, displacement=130.0, timestamp_sec=1.2, contributing_cameras=["cam_side"])
    assert ev3 is not None
    assert ev3.carton_count == 1
    assert ev3.direction == "IN"
    assert sm.total_counted == 1

    # Step 4: Subsequent frames while still after line -> MUST NEVER DOUBLE COUNT!
    for next_x in [540.0, 560.0, 600.0]:
        ev_dup = sm.update_carton_position("carton_A", coord=next_x, displacement=200.0, timestamp_sec=1.5, contributing_cameras=["cam_side"])
        assert ev_dup is None

    assert sm.total_counted == 1


def test_crossing_direction_reversal_handling():
    sm = CartonCrossingStateMachine(line_position_px=500.0, hysteresis_margin_px=20.0, required_direction="IN")

    # Starts after line (X = 600) and moves towards before line (X = 450) -> OUT direction
    sm.update_carton_position("carton_B", coord=600.0, displacement=5.0, timestamp_sec=1.0, contributing_cameras=["cam_side"])
    ev = sm.update_carton_position("carton_B", coord=450.0, displacement=150.0, timestamp_sec=1.5, contributing_cameras=["cam_side"])

    # Since required_direction is IN, an OUT reversal must not count!
    assert ev is None
    assert sm.total_counted == 0


# ===========================================================================
# 7. Consensus Policy Tests
# ===========================================================================
def test_three_camera_consensus_levels():
    policy = ThreeCameraConsensusPolicy()

    # 3-Camera Unanimous
    v3 = policy.evaluate(["cam_top", "cam_side", "cam_flank"], [0.9, 0.85, 0.88])
    assert v3.level == ConsensusLevel.THREE_CAMERA_UNANIMOUS
    assert v3.action == DecisionAction.COMMIT
    assert len(v3.missing_cameras) == 0

    # 2-Camera Agreement (e.g. flank occluded)
    v2 = policy.evaluate(["cam_top", "cam_side"], [0.9, 0.85])
    assert v2.level == ConsensusLevel.TWO_CAMERA_AGREEMENT
    assert v2.action == DecisionAction.COMMIT
    assert v2.missing_cameras == ["cam_flank"]

    # 1-Camera with High Confidence -> Hold for review
    v1_high = policy.evaluate(["cam_top"], [0.85])
    assert v1_high.level == ConsensusLevel.SINGLE_CAMERA_OBSERVATION
    assert v1_high.action == DecisionAction.HOLD_FOR_REVIEW

    # 1-Camera with Low Confidence -> Suppress as noise
    v1_low = policy.evaluate(["cam_top"], [0.35])
    assert v1_low.level == ConsensusLevel.SINGLE_CAMERA_OBSERVATION
    assert v1_low.action == DecisionAction.SUPPRESS_AS_NOISE


# ===========================================================================
# 8. Stack Disambiguation Tests
# ===========================================================================
def test_stack_disambiguator_scenarios():
    disambiguator = StackDisambiguator()

    # Single carton
    r_single = disambiguator.disambiguate({"cam_top": 1, "cam_side": 1, "cam_flank": 1})
    assert r_single.resolution_state == StackResolutionState.INDIVIDUALLY_RESOLVED
    assert r_single.resolved_count == 1
    assert not r_single.is_ambiguous

    # 2 cartons vertically stacked, verified by both side profiles
    r_stacked = disambiguator.disambiguate({"cam_top": 1, "cam_side": 2, "cam_flank": 2})
    assert r_stacked.resolution_state == StackResolutionState.INDIVIDUALLY_RESOLVED
    assert r_stacked.resolved_count == 2
    assert not r_stacked.is_ambiguous

    # Conflicting stack counts -> MUST NOT GUESS, mark AMBIGUOUS
    r_conflict = disambiguator.disambiguate({"cam_top": 1, "cam_side": 2, "cam_flank": 3})
    assert r_conflict.resolution_state == StackResolutionState.AMBIGUOUS_STACK
    assert r_conflict.is_ambiguous
    assert r_conflict.requires_human_review


# ===========================================================================
# 9. Phase 2 & 3: PerCameraTripwire and Independent Event Tests
# ===========================================================================
from apps.truck_loading_detection.three_camera_system.corridor import (
    CameraTripwireConfig,
    PerCameraTripwire,
    LocalCameraCrossingCandidate,
)
from apps.truck_loading_detection.three_camera_system.pipeline import CrossingWindow


class MockTracklet:
    def __init__(self, tid: str, center: Tuple[float, float], disp: float = 25.0, conf: float = 0.85):
        self.local_track_id = tid
        self.center = center
        self.displacement = disp
        self.current_confidence = conf
        self.width = 60.0
        self.height = 60.0
        self.current_bbox = (center[0]-30, center[1]-30, center[0]+30, center[1]+30)


def test_per_camera_tripwire_vertical():
    """Verify cam_side vertical tripwire triggers on rightward movement."""
    cfg = CameraTripwireConfig(
        camera_id="cam_side",
        orientation="vertical",
        line_pos_frac=0.50,
        direction_sign=1,
        min_displacement_px=20.0,
    )
    tw = PerCameraTripwire(cfg)

    # Frame dims: (600, 1000). Tripwire line at X = 0.50 * 1000 = 500.
    # Frame 1: X=300 -> before line
    trk1 = MockTracklet("trk_1", center=(300.0, 300.0), disp=5.0)
    c1 = tw.update(trk1, frame_idx=10, timestamp_sec=0.33, frame_dims=(600, 1000))
    assert c1 is None

    # Frame 2: X=450 -> still before line
    trk2 = MockTracklet("trk_1", center=(450.0, 300.0), disp=15.0)
    c2 = tw.update(trk2, frame_idx=11, timestamp_sec=0.36, frame_dims=(600, 1000))
    assert c2 is None

    # Frame 3: X=580 -> crossed tripwire line (500 + 40 hysteresis = 540) with disp=30 > min_disp 20 -> TRIGGER!
    trk3 = MockTracklet("trk_1", center=(580.0, 300.0), disp=30.0)
    c3 = tw.update(trk3, frame_idx=12, timestamp_sec=0.40, frame_dims=(600, 1000))
    assert c3 is not None
    assert c3.camera_id == "cam_side"
    assert c3.local_track_id == "trk_1"
    assert c3.direction == "IN"
    assert c3.confidence >= 0.85

    # Frame 4: X=650 -> MUST NOT DUPLICATE
    trk4 = MockTracklet("trk_1", center=(650.0, 300.0), disp=40.0)
    c4 = tw.update(trk4, frame_idx=13, timestamp_sec=0.43, frame_dims=(600, 1000))
    assert c4 is None


def test_per_camera_tripwire_horizontal_cam_top():
    """Verify cam_top horizontal tripwire triggers on downward movement across floor corridor."""
    cfg = CameraTripwireConfig(
        camera_id="cam_top",
        orientation="horizontal",
        line_pos_frac=0.72,
        direction_sign=1,
        min_displacement_px=15.0,
    )
    tw = PerCameraTripwire(cfg)

    # Frame shape: (848, 476). Tripwire line Y = 0.72 * 848 = 610.56
    # Frame 1: Y=400
    tw.update(MockTracklet("trk_top_1", center=(240.0, 400.0), disp=5.0), frame_idx=1, timestamp_sec=0.1, frame_dims=(848, 476))
    # Frame 2: Y=550
    tw.update(MockTracklet("trk_top_1", center=(240.0, 550.0), disp=15.0), frame_idx=2, timestamp_sec=0.2, frame_dims=(848, 476))
    # Frame 3: Y=670 (crossed 610.56 + hysteresis) with disp=35 -> TRIGGER!
    cand = tw.update(MockTracklet("trk_top_1", center=(240.0, 670.0), disp=35.0), frame_idx=3, timestamp_sec=0.3, frame_dims=(848, 476))

    assert cand is not None
    assert cand.camera_id == "cam_top"
    assert cand.direction == "IN"


def test_tripwire_reversal_rejection():
    """Verify reverse movement across tripwire is rejected."""
    cfg = CameraTripwireConfig(
        camera_id="cam_side",
        orientation="vertical",
        line_pos_frac=0.50,
        direction_sign=1,
        min_displacement_px=20.0,
    )
    tw = PerCameraTripwire(cfg)

    # Starts at right (X=700) and moves left to X=300
    tw.update(MockTracklet("trk_rev", center=(700.0, 300.0), disp=5.0), frame_idx=1, timestamp_sec=0.1, frame_dims=(600, 1000))
    cand = tw.update(MockTracklet("trk_rev", center=(300.0, 300.0), disp=40.0), frame_idx=2, timestamp_sec=0.2, frame_dims=(600, 1000))

    # Direction is OUT (init was after, now before)
    assert cand is not None
    assert cand.direction == "OUT"


def test_sliding_window_event_coalescence():
    """Verify events from cam_side and cam_flank within 0.85s are coalesced into a single physical crossing."""
    cand_side = LocalCameraCrossingCandidate(
        local_event_id="ev_side_1",
        camera_id="cam_side",
        local_track_id="trk_s1",
        timestamp_sec=2.10,
        direction="IN",
        confidence=0.88,
        bbox=(480, 200, 560, 280),
        aspect_ratio=1.0,
        estimated_qty=1,
        displacement=35.0,
    )
    cand_flank = LocalCameraCrossingCandidate(
        local_event_id="ev_flank_1",
        camera_id="cam_flank",
        local_track_id="trk_f1",
        timestamp_sec=2.45,  # within 0.35s < 0.85s window
        direction="IN",
        confidence=0.82,
        bbox=(490, 210, 570, 290),
        aspect_ratio=1.0,
        estimated_qty=1,
        displacement=30.0,
    )

    win = CrossingWindow(window_id="win_1", start_time=cand_side.timestamp_sec, last_time=cand_side.timestamp_sec, local_events={cand_side.camera_id: [cand_side]})

    # Check temporal proximity
    max_time_diff_sec = 0.85
    assert abs(cand_flank.timestamp_sec - win.start_time) <= max_time_diff_sec
    win.local_events.setdefault(cand_flank.camera_id, []).append(cand_flank)
    win.last_time = max(win.last_time, cand_flank.timestamp_sec)

    assert set(win.local_events.keys()) == {"cam_side", "cam_flank"}
    assert win.last_time - win.start_time == pytest.approx(0.35, rel=1e-3)

    # Verify a candidate 2.0s later is NOT associated (distinct carton)
    cand_late = LocalCameraCrossingCandidate(
        local_event_id="ev_late",
        camera_id="cam_side",
        local_track_id="trk_s2",
        timestamp_sec=4.50,
        direction="IN",
        confidence=0.85,
        bbox=(500, 200, 580, 280),
        aspect_ratio=1.0,
        estimated_qty=1,
        displacement=30.0,
    )
    assert abs(cand_late.timestamp_sec - win.start_time) > max_time_diff_sec


# ===========================================================================
# 10. Count Semantics & Event State Invariants
# ===========================================================================
from apps.truck_loading_detection.three_camera_system.pipeline import ThreeCameraCartonPipeline, FrameTickOutput


def test_count_semantics_confirmed_vs_review_required(monkeypatch):
    """Verify that HOLD_FOR_REVIEW does NOT increment confirmed_count, only review_required_count."""
    pipeline = ThreeCameraCartonPipeline(conf_threshold=0.20)

    assert pipeline.confirmed_count == 0
    assert pipeline.review_required_count == 0
    assert pipeline.provisional_count == 0

    # 1. Simulate single-camera candidate (cam_side only) -> HOLD_FOR_REVIEW
    cand_single = LocalCameraCrossingCandidate(
        local_event_id="ev_s1",
        camera_id="cam_side",
        local_track_id="trk_s1",
        timestamp_sec=1.0,
        direction="IN",
        confidence=0.35,
        bbox=(400, 200, 500, 300),
        aspect_ratio=1.0,
        estimated_qty=1,
    )
    win_single = CrossingWindow(
        window_id="win_s1",
        start_time=1.0,
        last_time=1.0,
        local_events={"cam_side": [cand_single]},
    )
    pipeline._active_crossing_windows.append(win_single)

    # Trigger evaluation at t = 2.5s (> window duration 1.2s)
    # Mocking frames
    empty_frames = {"cam_top": np.zeros((10, 10, 3), dtype=np.uint8), "cam_side": np.zeros((10, 10, 3), dtype=np.uint8), "cam_flank": np.zeros((10, 10, 3), dtype=np.uint8)}
    # Avoid real detector run
    monkeypatch.setattr(pipeline.detector, "detect", lambda *args, **kwargs: [])
    tick = pipeline.process_synchronized_frames(empty_frames, frame_idx=75)

    # Invariant: confirmed_count MUST be 0! review_required_count MUST be 1!
    assert pipeline.confirmed_count == 0
    assert pipeline.review_required_count == 1
    assert pipeline.provisional_count == 1
    assert tick.confirmed_count == 0
    assert tick.review_required_count == 1
    assert tick.provisional_count == 1

    # 2. Simulate 2-camera candidate (cam_side + cam_flank) -> COMMIT
    cand_s2 = LocalCameraCrossingCandidate(
        local_event_id="ev_s2", camera_id="cam_side", local_track_id="trk_s2",
        timestamp_sec=5.0, direction="IN", confidence=0.85, bbox=(400, 200, 500, 300), aspect_ratio=1.0, estimated_qty=1
    )
    cand_f2 = LocalCameraCrossingCandidate(
        local_event_id="ev_f2", camera_id="cam_flank", local_track_id="trk_f2",
        timestamp_sec=5.2, direction="IN", confidence=0.80, bbox=(420, 200, 520, 300), aspect_ratio=1.0, estimated_qty=1
    )
    win_multi = CrossingWindow(
        window_id="win_m2",
        start_time=5.0,
        last_time=5.2,
        local_events={"cam_side": [cand_s2], "cam_flank": [cand_f2]},
    )
    pipeline._active_crossing_windows.append(win_multi)

    tick2 = pipeline.process_synchronized_frames(empty_frames, frame_idx=200)

    # Now confirmed_count increments to 1, review_required remains 1, provisional becomes 2!
    assert pipeline.confirmed_count == 1
    assert pipeline.review_required_count == 1
    assert pipeline.provisional_count == 2
    assert tick2.confirmed_count == 1
    assert tick2.review_required_count == 1
    assert tick2.provisional_count == 2


def test_stack_disambiguator_3d_grid_resolution():
    """Verify that 3D grid stacks (N cols from top x H layers from side/flank) correctly resolve Total = N x H."""
    disambiguator = StackDisambiguator()

    # Scenario A: 2 columns in top view, 3 layers in side & flank views -> 2 x 3 = 6 cartons!
    counts_6 = {"cam_top": 2, "cam_side": 3, "cam_flank": 3}
    res_6 = disambiguator.disambiguate(counts_6)
    assert res_6.resolution_state == StackResolutionState.INDIVIDUALLY_RESOLVED
    assert res_6.resolved_count == 6
    assert not res_6.is_ambiguous
    assert not res_6.requires_human_review

    # Scenario B: 3 columns in top view, 2 layers in side & flank views -> 3 x 2 = 6 cartons!
    counts_3x2 = {"cam_top": 3, "cam_side": 2, "cam_flank": 2}
    res_3x2 = disambiguator.disambiguate(counts_3x2)
    assert res_3x2.resolution_state == StackResolutionState.INDIVIDUALLY_RESOLVED
    assert res_3x2.resolved_count == 6

    # Scenario C: 2 columns in top view, but side (3 layers) and flank (2 layers) disagree -> Flag Ambiguous
    counts_disagree = {"cam_top": 2, "cam_side": 3, "cam_flank": 2}
    res_disagree = disambiguator.disambiguate(counts_disagree)
    assert res_disagree.resolution_state == StackResolutionState.AMBIGUOUS_STACK
    assert res_disagree.is_ambiguous
    assert res_disagree.requires_human_review
    assert res_disagree.resolved_count == 6  # 2 cols * max(3, 2)


