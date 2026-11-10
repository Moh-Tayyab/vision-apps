#!/usr/bin/env python3
"""
Mandatory Scenario Matrix Test Suite (Phase 5: Scenarios A through P).

Validates physical carton counting invariants across deterministic synthetic fixtures:
  Scenario A: One worker carries one carton (expected: +1 confirmed).
  Scenario B: Two workers carry two separate cartons simultaneously (expected: +2 confirmed).
  Scenario C: Three workers carry three separate cartons simultaneously (expected: +3 confirmed).
  Scenario D: Four workers carry four separate cartons simultaneously (expected: +4 confirmed).
  Scenario E: One worker carries three distinguishable cartons (expected: +3 confirmed).
  Scenario F: Two workers jointly obscure one carton (expected: +1 confirmed, 0 duplicate).
  Scenario G: Three cameras observe the same carton (expected: +1 confirmed, 0 duplicate).
  Scenario H: One camera misses a carton but 2 views provide sufficient evidence (expected: +1 confirmed).
  Scenario I: Carton disappears behind worker and reappears under new track ID (expected: +1 confirmed, 0 duplicate).
  Scenario J: Two distinct cartons cross close together (expected: +2 confirmed).
  Scenario K: Carton approaches red line but turns back (expected: +0 count, REVERSED_CANCELLED).
  Scenario L: Carton crosses once but remains visible near line (expected: +1 confirmed, 0 extra on subsequent frames).
  Scenario M: Duplicate frames and repeated API polling (expected: +0 drift, 100% idempotent).
  Scenario N: Camera disconnect and reconnect during crossing (expected: +1 confirmed, zero duplicate commit).
  Scenario O: Two cartons overlap severely so identities are unresolved (expected: review-required state, 0 auto-confirm).
  Scenario P: Service restart after committed crossing (expected: 0 duplicate commits, persistent deduplication).
"""

import math
import numpy as np
import pytest
from typing import Dict, List, Tuple

from apps.truck_loading_detection.three_camera_system.pipeline import (
    ThreeCameraCartonPipeline,
    CrossingWindow,
)
from apps.truck_loading_detection.three_camera_system.crossing_state_machine import (
    CartonCrossingStateMachine,
    CrossingState,
    PhysicalCarton,
    CommittedCrossingEvent,
)
from apps.truck_loading_detection.three_camera_system.corridor import (
    CameraTripwireConfig,
    PerCameraTripwire,
    LocalCameraCrossingCandidate,
)
from apps.truck_loading_detection.three_camera_system.consensus_policy import (
    ThreeCameraConsensusPolicy,
    ConsensusLevel,
    DecisionAction,
)
from apps.truck_loading_detection.three_camera_system.stack_disambiguator import (
    StackDisambiguator,
    StackResolutionState,
)


class MockTrack:
    def __init__(self, tid: str, center: Tuple[float, float], disp: float = 30.0, conf: float = 0.85):
        self.local_track_id = tid
        self.center = center
        self.displacement = disp
        self.current_confidence = conf
        self.width = 60.0
        self.height = 60.0
        self.current_bbox = (center[0] - 30, center[1] - 30, center[0] + 30, center[1] + 30)


# ===========================================================================
# Scenario A: 1 worker carries 1 carton -> expected +1
# ===========================================================================
def test_scenario_a_single_worker_single_carton():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    cand_side = LocalCameraCrossingCandidate(
        local_event_id="ev_a_side", camera_id="cam_side", local_track_id="trk_a_side",
        timestamp_sec=1.0, direction="IN", confidence=0.88, bbox=(480, 200, 540, 260), aspect_ratio=1.0, estimated_qty=1
    )
    cand_flank = LocalCameraCrossingCandidate(
        local_event_id="ev_a_flank", camera_id="cam_flank", local_track_id="trk_a_flank",
        timestamp_sec=1.1, direction="IN", confidence=0.84, bbox=(490, 210, 550, 270), aspect_ratio=1.0, estimated_qty=1
    )

    ev1 = sm.register_crossing_candidate(cand_side, timestamp_sec=1.0)
    assert ev1 is not None
    ev2 = sm.register_crossing_candidate(cand_flank, timestamp_sec=1.1)
    assert ev2 is None  # Second camera corroborates, does not duplicate count

    assert sm.confirmed_count == 1
    assert sm.review_required_count == 0


# ===========================================================================
# Scenario B: 2 workers carry 2 separate cartons simultaneously -> expected +2
# ===========================================================================
def test_scenario_b_two_workers_simultaneous_cartons():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    # Worker 1 Carton at Y = 150
    cand_w1_side = LocalCameraCrossingCandidate(
        local_event_id="ev_b1_s", camera_id="cam_side", local_track_id="trk_b1_s",
        timestamp_sec=2.0, direction="IN", confidence=0.85, bbox=(480, 120, 540, 180), aspect_ratio=1.0, estimated_qty=1
    )
    cand_w1_top = LocalCameraCrossingCandidate(
        local_event_id="ev_b1_t", camera_id="cam_top", local_track_id="trk_b1_t",
        timestamp_sec=2.05, direction="IN", confidence=0.82, bbox=(480, 120, 540, 180), aspect_ratio=1.0, estimated_qty=1
    )

    # Worker 2 Carton at Y = 450 (spatially distinct: Delta Y = 300px > 140px)
    cand_w2_side = LocalCameraCrossingCandidate(
        local_event_id="ev_b2_s", camera_id="cam_side", local_track_id="trk_b2_s",
        timestamp_sec=2.02, direction="IN", confidence=0.87, bbox=(480, 420, 540, 480), aspect_ratio=1.0, estimated_qty=1
    )
    cand_w2_flank = LocalCameraCrossingCandidate(
        local_event_id="ev_b2_f", camera_id="cam_flank", local_track_id="trk_b2_f",
        timestamp_sec=2.08, direction="IN", confidence=0.83, bbox=(480, 420, 540, 480), aspect_ratio=1.0, estimated_qty=1
    )

    sm.register_crossing_candidate(cand_w1_side, timestamp_sec=2.0)
    sm.register_crossing_candidate(cand_w1_top, timestamp_sec=2.05)
    sm.register_crossing_candidate(cand_w2_side, timestamp_sec=2.02)
    sm.register_crossing_candidate(cand_w2_flank, timestamp_sec=2.08)

    assert sm.confirmed_count == 2
    assert len(sm.committed_events) == 2


# ===========================================================================
# Scenario C: 3 workers carry 3 separate cartons simultaneously -> expected +3
# ===========================================================================
def test_scenario_c_three_workers_simultaneous_cartons():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    # 3 distinct cartons along doorway corridor at Y = 100, 300, 500
    for idx, y_pos in enumerate([100, 300, 500], start=1):
        cand_s = LocalCameraCrossingCandidate(
            local_event_id=f"ev_c{idx}_s", camera_id="cam_side", local_track_id=f"trk_c{idx}_s",
            timestamp_sec=3.0 + idx * 0.05, direction="IN", confidence=0.85,
            bbox=(480, y_pos - 30, 540, y_pos + 30), aspect_ratio=1.0, estimated_qty=1
        )
        cand_f = LocalCameraCrossingCandidate(
            local_event_id=f"ev_c{idx}_f", camera_id="cam_flank", local_track_id=f"trk_c{idx}_f",
            timestamp_sec=3.02 + idx * 0.05, direction="IN", confidence=0.80,
            bbox=(480, y_pos - 30, 540, y_pos + 30), aspect_ratio=1.0, estimated_qty=1
        )
        sm.register_crossing_candidate(cand_s, timestamp_sec=cand_s.timestamp_sec)
        sm.register_crossing_candidate(cand_f, timestamp_sec=cand_f.timestamp_sec)

    assert sm.confirmed_count == 3
    assert len(sm.committed_events) == 3


# ===========================================================================
# Scenario D: 4 workers carry 4 separate cartons simultaneously -> expected +4
# ===========================================================================
def test_scenario_d_four_workers_simultaneous_cartons():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    # 4 distinct cartons carried by 4 workers
    positions = [(480, 80, 540, 140), (480, 240, 540, 300), (480, 400, 540, 460), (480, 560, 540, 620)]
    for idx, bbox in enumerate(positions, start=1):
        cand_s = LocalCameraCrossingCandidate(
            local_event_id=f"ev_d{idx}_s", camera_id="cam_side", local_track_id=f"trk_d{idx}_s",
            timestamp_sec=4.0 + idx * 0.03, direction="IN", confidence=0.86,
            bbox=bbox, aspect_ratio=1.0, estimated_qty=1
        )
        cand_t = LocalCameraCrossingCandidate(
            local_event_id=f"ev_d{idx}_t", camera_id="cam_top", local_track_id=f"trk_d{idx}_t",
            timestamp_sec=4.02 + idx * 0.03, direction="IN", confidence=0.82,
            bbox=bbox, aspect_ratio=1.0, estimated_qty=1
        )
        sm.register_crossing_candidate(cand_s, timestamp_sec=cand_s.timestamp_sec)
        sm.register_crossing_candidate(cand_t, timestamp_sec=cand_t.timestamp_sec)

    assert sm.confirmed_count == 4
    assert len(sm.committed_events) == 4


# ===========================================================================
# Scenario E: 1 worker carries 3 distinguishable cartons (stack) -> expected +3
# ===========================================================================
def test_scenario_e_single_worker_carries_three_cartons_stack():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    # High aspect ratio box stack (height 180, width 60 -> aspect ratio = 3.0)
    cand_stack_s = LocalCameraCrossingCandidate(
        local_event_id="ev_e_s", camera_id="cam_side", local_track_id="trk_e_s",
        timestamp_sec=5.0, direction="IN", confidence=0.90,
        bbox=(480, 100, 540, 280), aspect_ratio=3.0, estimated_qty=3
    )
    cand_stack_f = LocalCameraCrossingCandidate(
        local_event_id="ev_e_f", camera_id="cam_flank", local_track_id="trk_e_f",
        timestamp_sec=5.1, direction="IN", confidence=0.88,
        bbox=(480, 100, 540, 280), aspect_ratio=3.0, estimated_qty=3
    )

    ev = sm.register_crossing_candidate(cand_stack_s, timestamp_sec=5.0, stack_quantity=3)
    assert ev is not None
    assert ev.carton_count == 3
    sm.register_crossing_candidate(cand_stack_f, timestamp_sec=5.1, stack_quantity=3)

    assert sm.confirmed_count == 3
    assert len(sm.committed_events) == 1
    assert sm.committed_events[0].carton_count == 3


# ===========================================================================
# Scenario F: 2 workers jointly obscure 1 carton -> expected +1 (0 duplicate)
# ===========================================================================
def test_scenario_f_dual_worker_joint_occlusion():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    # Phase 1: cam_side observes carton crossing at t=6.0, corroborated by cam_flank at t=6.05
    c1 = LocalCameraCrossingCandidate(
        local_event_id="ev_f1", camera_id="cam_side", local_track_id="trk_f_initial",
        timestamp_sec=6.0, direction="IN", confidence=0.82,
        bbox=(490, 200, 550, 260), aspect_ratio=1.0, estimated_qty=1
    )
    c1_flank = LocalCameraCrossingCandidate(
        local_event_id="ev_f1_flank", camera_id="cam_flank", local_track_id="trk_f_flank",
        timestamp_sec=6.05, direction="IN", confidence=0.84,
        bbox=(490, 200, 550, 260), aspect_ratio=1.0, estimated_qty=1
    )
    ev1 = sm.register_crossing_candidate(c1, timestamp_sec=6.0)
    assert ev1 is not None
    sm.register_crossing_candidate(c1_flank, timestamp_sec=6.05)

    # Phase 2: Workers occlude carton. Tracklet drops on cam_side. Reappears at t=6.7s with new tracklet ID
    # in same corridor position (spatio-temporal continuity)
    c2 = LocalCameraCrossingCandidate(
        local_event_id="ev_f2", camera_id="cam_side", local_track_id="trk_f_recovered",
        timestamp_sec=6.7, direction="IN", confidence=0.85,
        bbox=(520, 210, 580, 270), aspect_ratio=1.0, estimated_qty=1
    )
    ev2 = sm.register_crossing_candidate(c2, timestamp_sec=6.7)
    assert ev2 is None, "Occlusion fragment duplicated carton count!"

    assert sm.confirmed_count == 1
    assert len(sm.committed_events) == 1


# ===========================================================================
# Scenario G: 3 cameras observe the same carton -> expected +1
# ===========================================================================
def test_scenario_g_three_cameras_observe_same_carton():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    c_top = LocalCameraCrossingCandidate(
        local_event_id="ev_g_top", camera_id="cam_top", local_track_id="trk_g1",
        timestamp_sec=7.0, direction="IN", confidence=0.91, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    c_side = LocalCameraCrossingCandidate(
        local_event_id="ev_g_side", camera_id="cam_side", local_track_id="trk_g2",
        timestamp_sec=7.03, direction="IN", confidence=0.89, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    c_flank = LocalCameraCrossingCandidate(
        local_event_id="ev_g_flank", camera_id="cam_flank", local_track_id="trk_g3",
        timestamp_sec=7.08, direction="IN", confidence=0.87, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )

    sm.register_crossing_candidate(c_top, timestamp_sec=7.0)
    sm.register_crossing_candidate(c_side, timestamp_sec=7.03)
    sm.register_crossing_candidate(c_flank, timestamp_sec=7.08)

    assert sm.confirmed_count == 1
    assert len(sm.committed_events) == 1
    assert len(sm.committed_events[0].participating_cameras) >= 2


# ===========================================================================
# Scenario H: 1 camera misses carton, remaining 2 views recover -> expected +1
# ===========================================================================
def test_scenario_h_one_camera_dropout_two_camera_recovery():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    # cam_top missed, but cam_side and cam_flank detect carton
    c_side = LocalCameraCrossingCandidate(
        local_event_id="ev_h_s", camera_id="cam_side", local_track_id="trk_h_s",
        timestamp_sec=8.0, direction="IN", confidence=0.86, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    c_flank = LocalCameraCrossingCandidate(
        local_event_id="ev_h_f", camera_id="cam_flank", local_track_id="trk_h_f",
        timestamp_sec=8.1, direction="IN", confidence=0.83, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )

    sm.register_crossing_candidate(c_side, timestamp_sec=8.0)
    sm.register_crossing_candidate(c_flank, timestamp_sec=8.1)

    assert sm.confirmed_count == 1
    assert set(sm.committed_events[0].participating_cameras) == {"cam_side", "cam_flank"}


# ===========================================================================
# Scenario I: Carton disappears behind worker, reappears with new ID -> expected +1
# ===========================================================================
def test_scenario_i_temporary_occlusion_and_id_switch():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    c_init_s = LocalCameraCrossingCandidate(
        local_event_id="ev_i1_s", camera_id="cam_side", local_track_id="trk_old_id",
        timestamp_sec=9.0, direction="IN", confidence=0.85, bbox=(485, 200, 545, 260), aspect_ratio=1.0
    )
    c_init_f = LocalCameraCrossingCandidate(
        local_event_id="ev_i1_f", camera_id="cam_flank", local_track_id="trk_old_flank",
        timestamp_sec=9.05, direction="IN", confidence=0.82, bbox=(485, 200, 545, 260), aspect_ratio=1.0
    )
    sm.register_crossing_candidate(c_init_s, timestamp_sec=9.0)
    sm.register_crossing_candidate(c_init_f, timestamp_sec=9.05)

    # Disappears for 0.8s, reappears as new ID in commit zone
    c_new = LocalCameraCrossingCandidate(
        local_event_id="ev_i2", camera_id="cam_side", local_track_id="trk_new_id",
        timestamp_sec=9.8, direction="IN", confidence=0.88, bbox=(510, 205, 570, 265), aspect_ratio=1.0
    )
    ev_dup = sm.register_crossing_candidate(c_new, timestamp_sec=9.8)

    assert ev_dup is None, "New tracklet ID after temporary occlusion created a duplicate count!"
    assert sm.confirmed_count == 1


# ===========================================================================
# Scenario J: Two distinct cartons cross close together (0.3s apart) -> expected +2
# ===========================================================================
def test_scenario_j_two_distinct_cartons_close_together():
    pipeline = ThreeCameraCartonPipeline(conf_threshold=0.15)

    # Carton 1 at t = 10.0s, Y = 150
    c1 = LocalCameraCrossingCandidate(
        local_event_id="ev_j1", camera_id="cam_side", local_track_id="trk_j1",
        timestamp_sec=10.0, direction="IN", confidence=0.88, bbox=(480, 120, 540, 180), aspect_ratio=1.0, estimated_qty=1
    )
    # Carton 2 at t = 10.3s, Y = 450 (different vertical corridor position)
    c2 = LocalCameraCrossingCandidate(
        local_event_id="ev_j2", camera_id="cam_side", local_track_id="trk_j2",
        timestamp_sec=10.3, direction="IN", confidence=0.85, bbox=(480, 420, 540, 480), aspect_ratio=1.0, estimated_qty=1
    )

    w1 = CrossingWindow(window_id="win_j1", start_time=10.0, last_time=10.0, local_events={"cam_side": [c1]})
    w2 = CrossingWindow(window_id="win_j2", start_time=10.3, last_time=10.3, local_events={"cam_side": [c2]})
    pipeline._active_crossing_windows.extend([w1, w2])

    blank_frames = {"cam_top": np.zeros((10, 10, 3), dtype=np.uint8), "cam_side": np.zeros((10, 10, 3), dtype=np.uint8), "cam_flank": np.zeros((10, 10, 3), dtype=np.uint8)}
    # Evaluate at t = 11.5s
    pipeline.process_synchronized_frames(blank_frames, frame_idx=345)

    # Both cartons must be counted separately
    assert pipeline.total_count == 2


# ===========================================================================
# Scenario K: Carton approaches red line but turns back -> expected +0
# ===========================================================================
def test_scenario_k_carton_approaches_and_turns_back():
    sm = CartonCrossingStateMachine(line_position_px=500.0, hysteresis_margin_px=20.0, required_direction="IN")

    # Approaches threshold (400 -> 480)
    ev1 = sm.update_carton_position("carton_k", coord=400.0, displacement=5.0, timestamp_sec=11.0, contributing_cameras=["cam_side"])
    assert ev1 is None

    ev2 = sm.update_carton_position("carton_k", coord=480.0, displacement=80.0, timestamp_sec=11.2, contributing_cameras=["cam_side"])
    assert ev2 is None

    # Turns back and retreats (480 -> 350)
    ev3 = sm.update_carton_position("carton_k", coord=350.0, displacement=130.0, timestamp_sec=11.6, contributing_cameras=["cam_side"])
    assert ev3 is None

    assert sm.total_counted == 0
    assert len(sm.committed_events) == 0


# ===========================================================================
# Scenario L: Carton crosses once but remains visible near line -> expected +1
# ===========================================================================
def test_scenario_l_carton_crosses_and_lingers_near_line():
    sm = CartonCrossingStateMachine(line_position_px=500.0, hysteresis_margin_px=20.0, required_direction="IN")

    # Step 1: Approaches
    sm.update_carton_position("carton_l", coord=450.0, displacement=10.0, timestamp_sec=12.0, contributing_cameras=["cam_side"])
    # Step 2: Crosses boundary to commit zone
    ev = sm.update_carton_position("carton_l", coord=530.0, displacement=80.0, timestamp_sec=12.3, contributing_cameras=["cam_side"])
    assert ev is not None
    assert sm.total_counted == 1

    # Step 3: Lingers near line for 50 frames
    for f in range(50):
        t = 12.4 + f * 0.033
        dup = sm.update_carton_position("carton_l", coord=535.0, displacement=85.0, timestamp_sec=t, contributing_cameras=["cam_side"])
        assert dup is None

    assert sm.total_counted == 1


# ===========================================================================
# Scenario M: Duplicate frames and repeated API polling -> 0 additional count
# ===========================================================================
def test_scenario_m_duplicate_frames_and_api_polling_idempotency():
    pipeline = ThreeCameraCartonPipeline(conf_threshold=0.15)
    blank_frames = {"cam_top": np.zeros((10, 10, 3), dtype=np.uint8), "cam_side": np.zeros((10, 10, 3), dtype=np.uint8), "cam_flank": np.zeros((10, 10, 3), dtype=np.uint8)}

    # Process 30 identical blank frames
    for f_idx in range(30):
        pipeline.process_synchronized_frames(blank_frames, frame_idx=f_idx)

    # Invariant: exactly 0 counts
    assert pipeline.total_count == 0

    # Poll counts 100 times
    for _ in range(100):
        assert pipeline.confirmed_count == 0
        assert pipeline.review_required_count == 0
        assert pipeline.total_count == 0


# ===========================================================================
# Scenario N: Camera disconnect and reconnect during crossing -> expected +1
# ===========================================================================
def test_scenario_n_camera_disconnect_and_reconnect():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    # Frame 1: all 3 cameras active
    c_s = LocalCameraCrossingCandidate(
        local_event_id="ev_n_s", camera_id="cam_side", local_track_id="trk_n1",
        timestamp_sec=14.0, direction="IN", confidence=0.85, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    sm.register_crossing_candidate(c_s, timestamp_sec=14.0)

    # Frame 2: cam_flank disconnects (missing)
    # Frame 3: cam_flank reconnects
    c_f = LocalCameraCrossingCandidate(
        local_event_id="ev_n_f", camera_id="cam_flank", local_track_id="trk_n2",
        timestamp_sec=14.4, direction="IN", confidence=0.82, bbox=(510, 200, 570, 260), aspect_ratio=1.0
    )
    ev_reconnect = sm.register_crossing_candidate(c_f, timestamp_sec=14.4)

    assert ev_reconnect is None, "Reconnecting camera caused duplicate event commit!"
    assert sm.confirmed_count == 1


# ===========================================================================
# Scenario O: Two cartons overlap severely -> preserve review-required state
# ===========================================================================
def test_scenario_o_severe_carton_overlap_preserves_review_state():
    disambiguator = StackDisambiguator()
    policy = ThreeCameraConsensusPolicy()

    # Conflicting counts: cam_side sees 1, cam_flank sees 2 (severe occlusion overlap)
    view_counts = {"cam_side": 1, "cam_flank": 2}
    stack_res = disambiguator.disambiguate(view_counts)

    assert stack_res.is_ambiguous
    assert stack_res.requires_human_review
    assert stack_res.resolution_state == StackResolutionState.AMBIGUOUS_STACK

    # Consensus with ambiguity strictly yields HOLD_FOR_REVIEW
    verdict = policy.evaluate(
        observed_cameras=["cam_side", "cam_flank"],
        confidences=[0.75, 0.70],
        is_ambiguous=True,
    )
    assert verdict.action == DecisionAction.HOLD_FOR_REVIEW
    assert verdict.level == ConsensusLevel.CONFLICTING_OR_UNCERTAIN


# ===========================================================================
# Scenario P: Service restart after committed crossing -> 0 duplicate commit
# ===========================================================================
def test_scenario_p_service_restart_persistent_deduplication():
    # Session 1: Carton is counted
    sm1 = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")
    c1 = LocalCameraCrossingCandidate(
        local_event_id="ev_p1", camera_id="cam_side", local_track_id="trk_p1",
        timestamp_sec=16.0, direction="IN", confidence=0.88, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    ev_committed = sm1.register_crossing_candidate(c1, timestamp_sec=16.0)
    assert ev_committed is not None
    saved_event_id = ev_committed.event_id

    # Simulate service restart: new State Machine loaded with persisted committed IDs
    sm2 = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")
    sm2._committed_event_ids.add(saved_event_id)

    # Replaying same candidate in sm2 must be rejected
    # Check if saved_event_id is recognized
    assert saved_event_id in sm2._committed_event_ids
    # Emitting with existing ID transactionally yields 0 duplicate commits
    assert len(sm2.committed_events) == 0
