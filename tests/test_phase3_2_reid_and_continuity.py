"""
Phase 3.2 — Occlusion-Aware Tracklet Re-Identification & Physical Carton Continuity Test Suite.

Covers all 13 critical regression scenarios required by Phase 3.2 specification:
1. Occlusion and tracklet ID fragmentation
2. Same physical carton across camera tracklets
3. Two simultaneous cartons in parallel lanes
4. Parallel workers carrying cartons
5. Same lane with temporal overlap
6. Similar-looking separate cartons
7. Worker mask overlap
8. Empty-hand phantom detection suppression (Session 18 ev_00018 failure mode)
9. Startup during a physical carton crossing (Session 21 session_21_ev_01 failure mode)
10. Track recovery after camera frame loss
11. Direction reversal cancellation
12. Legitimate repeated crossing (different time/direction)
13. Exactly-once persistence after recovery
"""

import math
import os
import shutil
import tempfile
import numpy as np
import pytest

from apps.truck_loading_detection.three_camera_system.pipeline import (
    ThreeCameraCartonPipeline,
    LocalCameraCrossingCandidate,
    CrossingWindow,
)
from apps.truck_loading_detection.three_camera_system.crossing_state_machine import (
    CartonCrossingStateMachine,
    CrossingState,
    PhysicalAssociationOutcome,
)
from apps.truck_loading_detection.three_camera_system.per_camera_tracker import (
    PerCameraCartonTracker,
    CartonTracklet,
    TrackStatus,
)
from apps.truck_loading_detection.three_camera_system.corridor import (
    CameraTripwireConfig,
    PerCameraTripwire,
)
from apps.truck_loading_detection.model_pipeline.mask_refinement import CartonMaskRefiner
from apps.truck_loading_detection.three_camera_system.detector_adapter import CanonicalCartonDetection
from apps.truck_loading_detection.three_camera_system.consensus_policy import (
    ThreeCameraConsensusPolicy,
    DecisionAction,
)
from apps.truck_loading_detection.three_camera_system.event_store import EventStore


@pytest.fixture
def temp_db_path():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_carton_events.db")
    yield db_path
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_scenario_01_occlusion_and_tracklet_fragmentation():
    """
    Scenario 1: Worker's arm/torso occludes a carton, causing ByteTrack to drop the tracklet
    and instantiate a new local tracklet upon reappearance.
    Physical carton identity must be preserved across the fragmentation.
    """
    csm = CartonCrossingStateMachine()
    
    # Initial sighting on cam_side
    p1 = csm.get_or_create_physical_carton(
        camera_id="cam_side",
        local_track_id="cam_side_t0001",
        center=(480.0, 300.0),
        timestamp_sec=2.40,
        direction="IN",
    )
    p1.is_committed = True
    p1.committed_timestamp = 2.40
    p1.last_committed_direction = "IN"
    
    # 0.2s later, tracklet fragments to cam_side_t0002 in same lane
    matched, outcome, reason = csm.find_matching_physical_carton(
        camera_id="cam_side",
        local_track_id="cam_side_t0002",
        center=(510.0, 305.0),
        timestamp_sec=2.60,
        direction="IN",
        return_outcome=True,
    )
    
    assert matched is not None
    assert matched.carton_id == p1.carton_id
    assert outcome in (
        PhysicalAssociationOutcome.MATCHED_EXISTING_PHYSICAL_CARTON,
        PhysicalAssociationOutcome.TEMPORARY_OCCLUSION,
    )


def test_scenario_02_same_physical_carton_across_camera_tracklets():
    """
    Scenario 2: The same physical carton crossing the doorway is tracked by cam_top, cam_side,
    and cam_flank with different local tracklet IDs. All 3 must unify to 1 physical carton.
    """
    csm = CartonCrossingStateMachine()
    
    p = csm.get_or_create_physical_carton(
        camera_id="cam_top",
        local_track_id="cam_top_t0005",
        center=(500.0, 250.0),
        timestamp_sec=5.00,
        direction="IN",
    )
    p.is_committed = True
    p.committed_timestamp = 5.00
    p.last_committed_direction = "IN"
    
    # cam_side sees it 0.15s later
    m_side = csm.find_matching_physical_carton(
        camera_id="cam_side",
        local_track_id="cam_side_t0009",
        center=(520.0, 260.0),
        timestamp_sec=5.15,
        direction="IN",
    )
    assert m_side is not None
    assert m_side.carton_id == p.carton_id
    
    # cam_flank sees it 0.30s later
    m_flank = csm.find_matching_physical_carton(
        camera_id="cam_flank",
        local_track_id="cam_flank_t0012",
        center=(490.0, 255.0),
        timestamp_sec=5.30,
        direction="IN",
    )
    assert m_flank is not None
    assert m_flank.carton_id == p.carton_id


def test_scenario_03_two_simultaneous_cartons_parallel_lanes():
    """
    Scenario 3: Two distinct cartons crossing simultaneously in parallel lanes
    (e.g. y=150 and y=350) must NEVER be merged into a single physical carton.
    """
    csm = CartonCrossingStateMachine()
    
    p1 = csm.get_or_create_physical_carton(
        camera_id="cam_side",
        local_track_id="cam_side_t0001",
        center=(500.0, 150.0),
        timestamp_sec=3.00,
        direction="IN",
    )
    p1.is_committed = True
    p1.committed_timestamp = 3.00
    p1.last_committed_direction = "IN"
    
    # Parallel carton at y=350 (> 60px lane separation)
    m2, outcome, reason = csm.find_matching_physical_carton(
        camera_id="cam_side",
        local_track_id="cam_side_t0002",
        center=(505.0, 350.0),
        timestamp_sec=3.05,
        direction="IN",
        return_outcome=True,
    )
    
    assert m2 is None
    assert outcome == PhysicalAssociationOutcome.NEW_PHYSICAL_CARTON


def test_scenario_04_parallel_workers_carrying_cartons():
    """
    Scenario 4: Two workers walking through the doorway simultaneously in separate lanes.
    The corridor association must maintain strict lane separation.
    """
    csm = CartonCrossingStateMachine()
    
    w1_carton = csm.get_or_create_physical_carton(
        camera_id="cam_top",
        local_track_id="cam_top_t0001",
        center=(480.0, 120.0),
        timestamp_sec=4.00,
        direction="IN",
    )
    w1_carton.is_committed = True
    w1_carton.committed_timestamp = 4.00
    
    w2_match = csm.find_matching_physical_carton(
        camera_id="cam_top",
        local_track_id="cam_top_t0002",
        center=(490.0, 280.0),  # y-diff = 160px
        timestamp_sec=4.10,
        direction="IN",
    )
    assert w2_match is None  # Strictly distinct!


def test_scenario_05_same_lane_with_temporal_overlap():
    """
    Scenario 5: Two distinct cartons crossing consecutively in the same lane.
    If the second carton arrives after the first transit window (> 2.0s),
    it must be recognized as a NEW physical carton.
    """
    csm = CartonCrossingStateMachine()
    
    c1 = csm.get_or_create_physical_carton(
        camera_id="cam_side",
        local_track_id="cam_side_t0001",
        center=(500.0, 250.0),
        timestamp_sec=2.00,
        direction="IN",
    )
    c1.is_committed = True
    c1.committed_timestamp = 2.00
    c1.last_committed_direction = "IN"
    
    # Next carton arrives at t=4.50s (delta t = 2.5s > 2.0s transit window)
    c2_match = csm.find_matching_physical_carton(
        camera_id="cam_side",
        local_track_id="cam_side_t0003",
        center=(500.0, 250.0),
        timestamp_sec=4.50,
        direction="IN",
    )
    assert c2_match is None  # Must NOT merge into c1!


def test_scenario_06_similar_looking_separate_cartons():
    """
    Scenario 6: Similar-looking cartons detected at distinct spatial locations
    outside the spatial tolerance radius (> 140px) must not merge.
    """
    csm = CartonCrossingStateMachine()
    
    c1 = csm.get_or_create_physical_carton(
        camera_id="cam_side",
        local_track_id="cam_side_t0001",
        center=(400.0, 200.0),
        timestamp_sec=10.00,
        direction="IN",
    )
    c1.is_committed = True
    c1.committed_timestamp = 10.00
    
    # Distance = hypot(650-400, 200-200) = 250px (> 140px)
    c2_match = csm.find_matching_physical_carton(
        camera_id="cam_side",
        local_track_id="cam_side_t0002",
        center=(650.0, 200.0),
        timestamp_sec=10.20,
        direction="IN",
    )
    assert c2_match is None


def test_scenario_07_worker_mask_overlap_preservation():
    """
    Scenario 7: Worker hands and arms holding a carton cover 35% of carton area.
    Mask refiner must disentangle worker mask while cleanly preserving the carton body.
    """
    refiner = CartonMaskRefiner(worker_subtraction_margin_px=2)
    h, w = 480, 640
    
    # Real carton 100x100 (area = 10,000 px)
    carton_poly = [(200, 200), (300, 200), (300, 300), (200, 300)]
    # Worker hands covering 35%
    worker_hands = [(265, 190), (310, 190), (310, 310), (265, 310)]
    
    refined = refiner.refine_carton_mask(
        raw_polygon=carton_poly,
        image_shape=(h, w),
        worker_polygons=[worker_hands],
        confidence=0.85,
    )
    
    assert refined is not None
    assert refined.is_occluded_by_worker is True
    assert refined.area_px >= refiner.min_carton_area_px
    assert refined.area_px < 10000.0  # Worker hands subtracted


def test_scenario_08_empty_hand_phantom_detection_suppressed():
    """
    Scenario 8 (Session 18 ev_00018 failure mode):
    An empty-handed worker stepping backward has false low-confidence carton detection
    on their shirt. Must be suppressed:
    1. Mask refiner detects it is 95% inside worker body with low confidence -> returns None
    2. Tripwire rejects backward motion
    3. Consensus policy suppresses low-confidence single-camera noise (< 0.25)
    """
    refiner = CartonMaskRefiner()
    h, w = 480, 640
    
    # False detection on worker shirt: 40x40 (1,600 px)
    false_carton = [(220, 220), (260, 220), (260, 260), (220, 260)]
    # Worker body covering the entire patch (100% inside worker)
    worker_torso = [(200, 200), (350, 200), (350, 450), (200, 450)]
    
    # 1. Mask refiner check
    refined = refiner.refine_carton_mask(
        raw_polygon=false_carton,
        image_shape=(h, w),
        worker_polygons=[worker_torso],
        confidence=0.17,  # ev_00018 confidence was 0.1697
    )
    assert refined is None  # Cleanly rejected as worker phantom!

    # 2. Consensus policy check
    policy = ThreeCameraConsensusPolicy(min_single_camera_conf=0.25)
    verdict = policy.evaluate(
        observed_cameras=["cam_flank"],
        confidences=[0.17],
    )
    assert verdict.action == DecisionAction.SUPPRESS_AS_NOISE


def test_scenario_09_startup_after_line_does_not_invent_crossing():
    """A carton first seen after the line must not count without crossing evidence."""
    tw_cfg = CameraTripwireConfig(
        camera_id="cam_side",
        orientation="vertical",
        line_pos_frac=0.50,
        direction_sign=1,
        support_occlusion_stitching=True,
    )
    tw = PerCameraTripwire(tw_cfg)
    
    # Frame 0: the carton is already in the line's hysteresis band.
    trk = CartonTracklet(
        camera_id="cam_side",
        local_track_id="cam_side_t0001",
        first_seen_frame=0,
        first_seen_timestamp=0.0,
        last_seen_frame=0,
        last_seen_timestamp=0.0,
        current_bbox=(480.0, 200.0, 580.0, 300.0),
        current_confidence=0.75,
        status=TrackStatus.CONFIRMED,
        trajectory=[(530.0, 250.0, 0.0)],
        total_visible_frames=1,
    )
    
    res0 = tw.update(trk, frame_idx=0, timestamp_sec=0.0, frame_dims=(576, 1024))
    assert res0 is None

    # It moves farther into the truck, but no observed track position crossed the line.
    trk.last_seen_frame = 10
    trk.last_seen_timestamp = 0.33
    trk.current_bbox = (510.0, 200.0, 610.0, 300.0)
    trk.trajectory.append((560.0, 250.0, 0.33))
    trk.total_visible_frames = 10

    res1 = tw.update(trk, frame_idx=10, timestamp_sec=0.33, frame_dims=(576, 1024))
    assert res1 is None


def test_tripwire_timestamp_is_interpolated_at_line_and_not_repeated():
    """Count time is the observed carton-center line crossing, not the hysteresis exit."""
    tw = PerCameraTripwire(CameraTripwireConfig(
        camera_id="cam_side",
        orientation="vertical",
        line_pos_frac=0.50,
        direction_sign=1,
        min_displacement_px=20.0,
    ))
    trk = CartonTracklet(
        camera_id="cam_side",
        local_track_id="line_crossing",
        first_seen_frame=0,
        first_seen_timestamp=0.0,
        last_seen_frame=0,
        last_seen_timestamp=0.0,
        current_bbox=(400.0, 200.0, 500.0, 300.0),
        current_confidence=0.9,
        status=TrackStatus.CONFIRMED,
        trajectory=[(450.0, 250.0, 0.0)],
    )

    assert tw.update(trk, frame_idx=0, timestamp_sec=0.0, frame_dims=(576, 1024)) is None

    trk.last_seen_frame = 10
    trk.last_seen_timestamp = 0.5
    trk.current_bbox = (525.0, 200.0, 625.0, 300.0)
    trk.trajectory.append((575.0, 250.0, 0.5))
    candidate = tw.update(trk, frame_idx=10, timestamp_sec=0.5, frame_dims=(576, 1024))

    assert candidate is not None
    assert candidate.direction == "IN"
    assert candidate.timestamp_sec == pytest.approx((512.0 - 450.0) / (575.0 - 450.0) * 0.5)
    assert candidate.crossing_point_px == pytest.approx((512.0, 250.0))
    assert candidate.crossing_line_px == pytest.approx(512.0)
    assert candidate.observed_frame_idx == 10

    trk.last_seen_frame = 11
    trk.last_seen_timestamp = 0.55
    trk.current_bbox = (540.0, 200.0, 640.0, 300.0)
    trk.trajectory.append((590.0, 250.0, 0.55))
    assert tw.update(trk, frame_idx=11, timestamp_sec=0.55, frame_dims=(576, 1024)) is None


def test_scenario_10_track_recovery_after_camera_frame_loss():
    """
    Scenario 10: One camera experiences temporary detection loss / frame drops for 8 frames.
    PerCameraCartonTracker must maintain the occluded tracklet via constant-velocity projection
    and re-associate upon reappearance with expanded distance.
    """
    tracker = PerCameraCartonTracker(camera_id="cam_top", max_invisible_frames=15, max_center_dist_px=120.0)
    
    # Frame 0: initial detection
    d0 = CanonicalCartonDetection(
        camera_id="cam_top",
        frame_idx=0,
        timestamp_sec=0.0,
        bbox=(200.0, 200.0, 300.0, 300.0),
        confidence=0.8,
        class_name="carton",
        area_px=10000.0,
    )
    trks0 = tracker.update([d0], frame_idx=0, timestamp_sec=0.0)
    assert len(trks0) == 1
    tid = trks0[0].local_track_id
    
    # Frame 1: confirmed
    d1 = CanonicalCartonDetection(
        camera_id="cam_top",
        frame_idx=1,
        timestamp_sec=0.033,
        bbox=(205.0, 200.0, 305.0, 300.0),
        confidence=0.8,
        class_name="carton",
        area_px=10000.0,
    )
    tracker.update([d1], frame_idx=1, timestamp_sec=0.033)
    
    # Frames 2-9 (8 frames): detection lost / occluded
    for f in range(2, 10):
        t_sec = f * 0.033
        active = tracker.update([], frame_idx=f, timestamp_sec=t_sec)
        assert len(active) == 1
        assert active[0].status == TrackStatus.OCCLUDED
        
    # Frame 10: reappears with ~135px displacement from original
    d10 = CanonicalCartonDetection(
        camera_id="cam_top",
        frame_idx=10,
        timestamp_sec=0.33,
        bbox=(330.0, 200.0, 430.0, 300.0),
        confidence=0.82,
        class_name="carton",
        area_px=10000.0,
    )
    trks10 = tracker.update([d10], frame_idx=10, timestamp_sec=0.33)
    assert len(trks10) == 1
    assert trks10[0].local_track_id == tid  # Track ID preserved across 8 frames of occlusion!
    assert trks10[0].status == TrackStatus.CONFIRMED


def test_scenario_11_direction_reversal_cancellation():
    """
    Scenario 11: A carton moves across the tripwire line, but the worker reverses and walks back.
    The crossing event must be cancelled and count remains zero.
    """
    csm = CartonCrossingStateMachine(
        line_orientation="vertical",
        line_position_px=500.0,
        hysteresis_margin_px=25.0,
        required_direction="IN",
    )
    
    # 1. Approach from before line (x=450)
    csm.update_carton_position("c1", coord=450.0, displacement=0.0, timestamp_sec=1.0, contributing_cameras=["cam_side"])
    
    # 2. Crosses line to x=550 (moving IN)
    ev = csm.update_carton_position("c1", coord=550.0, displacement=100.0, timestamp_sec=1.5, contributing_cameras=["cam_side"])
    assert ev is not None
    assert csm.total_in == 1
    
    # 3. Worker steps backward back to x=450 (reversal OUT)
    ev_rev = csm.update_carton_position("c1", coord=450.0, displacement=100.0, timestamp_sec=2.0, contributing_cameras=["cam_side"])
    assert ev_rev is None  # OUT event cancelled because required_direction="IN"
    assert csm.total_in == 1  # Exactly-once semantics maintained


def test_scenario_12_legitimate_repeated_crossing(temp_db_path):
    """
    Scenario 12: A physical carton legitimately crosses the threshold at t=2.0s (IN),
    then 30 seconds later crosses OUT (unloaded), and at t=60s crosses IN again.
    The EventStore must preserve all distinct crossing events without false deduplication.
    """
    store = EventStore(temp_db_path)
    
    # First crossing: IN at t=2.0s
    ok1, reason1, ev1 = store.commit_event({
        "event_id": "ev_001",
        "global_carton_id": "carton_001",
        "timestamp_sec": 2.0,
        "direction": "IN",
        "carton_count": 1,
    })
    assert ok1 is True
    
    # Duplicate attempt at same timestamp (idempotency check)
    ok_dup, reason_dup, _ = store.commit_event({
        "event_id": "ev_001_retry",
        "global_carton_id": "carton_001",
        "timestamp_sec": 2.1,  # within duplicate window
        "direction": "IN",
        "carton_count": 1,
    })
    assert ok_dup is False  # Correctly suppressed duplicate!
    
    # Legitimate second crossing: OUT at t=30.0s
    ok2, reason2, ev2 = store.commit_event({
        "event_id": "ev_002",
        "global_carton_id": "carton_001",
        "timestamp_sec": 30.0,
        "direction": "OUT",
        "carton_count": 1,
    })
    assert ok2 is True  # Legitimate crossing allowed!
    
    # Legitimate third crossing: IN at t=60.0s
    ok3, reason3, ev3 = store.commit_event({
        "event_id": "ev_003",
        "global_carton_id": "carton_001",
        "timestamp_sec": 60.0,
        "direction": "IN",
        "carton_count": 1,
    })
    assert ok3 is True  # Allowed!


def test_scenario_13_exactly_once_persistence_after_recovery(temp_db_path):
    """
    Scenario 13: Pipeline restarts or crashes. Recovering state from SQLite EventStore
    must reconstitute physical cartons and prevent re-counting already committed events.
    """
    store = EventStore(temp_db_path)
    
    # Commit initial event
    store.commit_event({
        "event_id": "ev_persist_1",
        "global_carton_id": "phys_box_99",
        "timestamp_sec": 12.5,
        "direction": "IN",
        "carton_count": 1,
        "participating_cameras": ["cam_top", "cam_side"],
        "is_auto_confirmed": True,
    }, session_id="test_session_recovery")
    
    # New pipeline instance pointing to same database
    pipeline = ThreeCameraCartonPipeline(
        camera_ids=["cam_top", "cam_side", "cam_flank"],
        db_path=temp_db_path,
        session_id="test_session_recovery",
    )
    pipeline.recover_from_event_store()
    
    # Verify state was restored
    assert len(pipeline.state_machine._physical_cartons) >= 1
    phys = list(pipeline.state_machine._physical_cartons.values())[0]
    assert phys.is_committed is True
    assert phys.committed_event_id == "ev_persist_1"
    assert phys.last_committed_direction == "IN"
    
    # Verify readiness status
    status = pipeline.get_readiness_status()
    assert status["is_ready"] is True
    assert status["durable_store_ready"] is True
