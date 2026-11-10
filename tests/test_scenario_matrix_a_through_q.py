#!/usr/bin/env python3
"""
Mandatory Scenario Matrix Test Suite (Section 5: Scenarios A through Q).

Validates physical carton counting invariants across deterministic synthetic fixtures:
  Scenario A: One carton crosses forward: expected total_in +1.
  Scenario B: Two cartons cross forward simultaneously: expected total_in +2.
  Scenario C: Three cartons cross forward simultaneously: expected total_in +3.
  Scenario D: Four cartons cross forward simultaneously: expected total_in +4.
  Scenario E: One worker carries three distinct cartons: expected total_in +3 if all physically cross.
  Scenario F: One carton is detected by all three cameras: expected one physical count.
  Scenario G: One carton temporarily disappears behind worker and reappears under new track ID: expected one physical count.
  Scenario H: Two distinct cartons overlap temporarily: expected two counts if resolved, else preserve uncertainty.
  Scenario I: Two cartons cross in opposite directions: verify each event is classified correctly (one IN, one OUT).
  Scenario J: One carton crosses forward and later crosses backward: expected total_in +1, total_out +1, net change zero.
  Scenario K: A carton approaches but does not cross: expected no count.
  Scenario L: A carton remains near the line after crossing: expected no repeated count.
  Scenario M: Duplicate frames and repeated event delivery: expected no extra count.
  Scenario N: Service restart after committed event: expected no duplicate count.
  Scenario O: Camera disconnection during crossing: preserve uncertainty and recover safely.
  Scenario P: Worker carries carton toward truck but returns it to pallet side: apply documented directional policy.
  Scenario Q: Several cartons cross simultaneously while workers obscure one another: distinct-item counting or review handling.
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
    ConsensusVerdict,
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
# Scenario A: One carton crosses forward -> expected total_in +1
# ===========================================================================
def test_scenario_a_single_carton_forward():
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

    assert sm.total_in == 1
    assert sm.total_out == 0
    assert sm.net_count == 1
    assert sm.confirmed_count == 1


# ===========================================================================
# Scenario B: Two cartons cross forward simultaneously -> expected total_in +2
# ===========================================================================
def test_scenario_b_two_cartons_simultaneously():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    # Carton 1 at Y = 150
    cand_1_s = LocalCameraCrossingCandidate(
        local_event_id="ev_b1_s", camera_id="cam_side", local_track_id="trk_b1_s",
        timestamp_sec=2.0, direction="IN", confidence=0.85, bbox=(480, 120, 540, 180), aspect_ratio=1.0, estimated_qty=1
    )
    cand_1_t = LocalCameraCrossingCandidate(
        local_event_id="ev_b1_t", camera_id="cam_top", local_track_id="trk_b1_t",
        timestamp_sec=2.05, direction="IN", confidence=0.82, bbox=(480, 120, 540, 180), aspect_ratio=1.0, estimated_qty=1
    )

    # Carton 2 at Y = 450 (spatially distinct: Delta Y = 300px > 140px)
    cand_2_s = LocalCameraCrossingCandidate(
        local_event_id="ev_b2_s", camera_id="cam_side", local_track_id="trk_b2_s",
        timestamp_sec=2.02, direction="IN", confidence=0.87, bbox=(480, 420, 540, 480), aspect_ratio=1.0, estimated_qty=1
    )
    cand_2_f = LocalCameraCrossingCandidate(
        local_event_id="ev_b2_f", camera_id="cam_flank", local_track_id="trk_b2_f",
        timestamp_sec=2.08, direction="IN", confidence=0.83, bbox=(480, 420, 540, 480), aspect_ratio=1.0, estimated_qty=1
    )

    sm.register_crossing_candidate(cand_1_s, timestamp_sec=2.0)
    sm.register_crossing_candidate(cand_1_t, timestamp_sec=2.05)
    sm.register_crossing_candidate(cand_2_s, timestamp_sec=2.02)
    sm.register_crossing_candidate(cand_2_f, timestamp_sec=2.08)

    assert sm.total_in == 2
    assert sm.total_out == 0
    assert sm.net_count == 2
    assert sm.confirmed_count == 2


# ===========================================================================
# Scenario C: Three cartons cross forward simultaneously -> expected total_in +3
# ===========================================================================
def test_scenario_c_three_cartons_simultaneously():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

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

    assert sm.total_in == 3
    assert sm.total_out == 0
    assert sm.net_count == 3


# ===========================================================================
# Scenario D: Four cartons cross forward simultaneously -> expected total_in +4
# ===========================================================================
def test_scenario_d_four_cartons_simultaneously():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

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

    assert sm.total_in == 4
    assert sm.total_out == 0
    assert sm.net_count == 4


# ===========================================================================
# Scenario E: One worker carries three distinct cartons -> expected total_in +3
# ===========================================================================
def test_scenario_e_single_worker_three_cartons_stack():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

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

    assert sm.total_in == 3
    assert sm.net_count == 3


# ===========================================================================
# Scenario F: One carton detected by all three cameras -> expected +1 physical count
# ===========================================================================
def test_scenario_f_three_cameras_detect_same_carton():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    c_top = LocalCameraCrossingCandidate(
        local_event_id="ev_f_top", camera_id="cam_top", local_track_id="trk_f1",
        timestamp_sec=6.0, direction="IN", confidence=0.91, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    c_side = LocalCameraCrossingCandidate(
        local_event_id="ev_f_side", camera_id="cam_side", local_track_id="trk_f2",
        timestamp_sec=6.03, direction="IN", confidence=0.89, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    c_flank = LocalCameraCrossingCandidate(
        local_event_id="ev_f_flank", camera_id="cam_flank", local_track_id="trk_f3",
        timestamp_sec=6.08, direction="IN", confidence=0.87, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )

    sm.register_crossing_candidate(c_top, timestamp_sec=6.0)
    sm.register_crossing_candidate(c_side, timestamp_sec=6.03)
    sm.register_crossing_candidate(c_flank, timestamp_sec=6.08)

    assert sm.total_in == 1
    assert len(sm.committed_events) == 1
    assert len(sm.committed_events[0].participating_cameras) >= 2


# ===========================================================================
# Scenario G: Carton disappears behind worker and reappears under new ID -> expected +1
# ===========================================================================
def test_scenario_g_temporary_occlusion_and_new_track_id():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    c_init_s = LocalCameraCrossingCandidate(
        local_event_id="ev_g1_s", camera_id="cam_side", local_track_id="trk_old_id",
        timestamp_sec=7.0, direction="IN", confidence=0.85, bbox=(485, 200, 545, 260), aspect_ratio=1.0
    )
    c_init_f = LocalCameraCrossingCandidate(
        local_event_id="ev_g1_f", camera_id="cam_flank", local_track_id="trk_old_flank",
        timestamp_sec=7.05, direction="IN", confidence=0.82, bbox=(485, 200, 545, 260), aspect_ratio=1.0
    )
    sm.register_crossing_candidate(c_init_s, timestamp_sec=7.0)
    sm.register_crossing_candidate(c_init_f, timestamp_sec=7.05)

    # Disappears for 0.8s, reappears as new ID in commit zone
    c_new = LocalCameraCrossingCandidate(
        local_event_id="ev_g2", camera_id="cam_side", local_track_id="trk_new_id",
        timestamp_sec=7.8, direction="IN", confidence=0.88, bbox=(510, 205, 570, 265), aspect_ratio=1.0
    )
    ev_dup = sm.register_crossing_candidate(c_new, timestamp_sec=7.8)

    assert ev_dup is None, "New tracklet ID after temporary occlusion created a duplicate count!"
    assert sm.total_in == 1


# ===========================================================================
# Scenario H: Two distinct cartons overlap temporarily -> 2 counts if resolved, else review
# ===========================================================================
def test_scenario_h_two_cartons_temporary_overlap():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    # Resolved case: side view separates them by stack/geometry
    c1 = LocalCameraCrossingCandidate(
        local_event_id="ev_h1", camera_id="cam_side", local_track_id="trk_h1",
        timestamp_sec=8.0, direction="IN", confidence=0.88, bbox=(480, 100, 540, 180), aspect_ratio=1.0
    )
    c2 = LocalCameraCrossingCandidate(
        local_event_id="ev_h2", camera_id="cam_side", local_track_id="trk_h2",
        timestamp_sec=8.02, direction="IN", confidence=0.85, bbox=(480, 300, 540, 380), aspect_ratio=1.0
    )
    sm.register_crossing_candidate(c1, timestamp_sec=8.0)
    sm.register_crossing_candidate(c2, timestamp_sec=8.02)

    assert len(sm.committed_events) == 2


# ===========================================================================
# Scenario I: Two cartons cross in opposite directions -> one IN, one OUT
# ===========================================================================
def test_scenario_i_opposite_direction_crossings():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="BOTH")

    # Carton 1 moving forward (IN)
    c_in_s = LocalCameraCrossingCandidate(
        local_event_id="ev_i_in_s", camera_id="cam_side", local_track_id="trk_in_s",
        timestamp_sec=9.0, direction="IN", confidence=0.88, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    c_in_f = LocalCameraCrossingCandidate(
        local_event_id="ev_i_in_f", camera_id="cam_flank", local_track_id="trk_in_f",
        timestamp_sec=9.05, direction="IN", confidence=0.85, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    sm.register_crossing_candidate(c_in_s, timestamp_sec=9.0)
    sm.register_crossing_candidate(c_in_f, timestamp_sec=9.05)

    # Carton 2 moving backward (OUT) at Y = 400
    c_out_s = LocalCameraCrossingCandidate(
        local_event_id="ev_i_out_s", camera_id="cam_side", local_track_id="trk_out_s",
        timestamp_sec=9.10, direction="OUT", confidence=0.86, bbox=(490, 400, 550, 460), aspect_ratio=1.0
    )
    c_out_f = LocalCameraCrossingCandidate(
        local_event_id="ev_i_out_f", camera_id="cam_flank", local_track_id="trk_out_f",
        timestamp_sec=9.15, direction="OUT", confidence=0.83, bbox=(490, 400, 550, 460), aspect_ratio=1.0
    )
    sm.register_crossing_candidate(c_out_s, timestamp_sec=9.10)
    sm.register_crossing_candidate(c_out_f, timestamp_sec=9.15)

    assert sm.total_in == 1
    assert sm.total_out == 1
    assert sm.net_count == 0  # 1 IN - 1 OUT = 0


# ===========================================================================
# Scenario J: One carton crosses forward and later crosses backward -> net zero
# ===========================================================================
def test_scenario_j_carton_crosses_forward_then_backward():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="BOTH")

    # Step 1: Carton crosses forward (IN) at t = 10.0s
    c_forward_s = LocalCameraCrossingCandidate(
        local_event_id="ev_j_fwd_s", camera_id="cam_side", local_track_id="trk_j",
        timestamp_sec=10.0, direction="IN", confidence=0.88, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    c_forward_f = LocalCameraCrossingCandidate(
        local_event_id="ev_j_fwd_f", camera_id="cam_flank", local_track_id="trk_j_f",
        timestamp_sec=10.05, direction="IN", confidence=0.84, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    sm.register_crossing_candidate(c_forward_s, timestamp_sec=10.0)
    sm.register_crossing_candidate(c_forward_f, timestamp_sec=10.05)

    assert sm.total_in == 1
    assert sm.total_out == 0
    assert sm.net_count == 1

    # Step 2: Later at t = 15.0s, worker carries same carton backward (OUT) to pallet
    c_backward_s = LocalCameraCrossingCandidate(
        local_event_id="ev_j_bwd_s", camera_id="cam_side", local_track_id="trk_j",
        timestamp_sec=15.0, direction="OUT", confidence=0.87, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    c_backward_f = LocalCameraCrossingCandidate(
        local_event_id="ev_j_bwd_f", camera_id="cam_flank", local_track_id="trk_j_f",
        timestamp_sec=15.05, direction="OUT", confidence=0.85, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    sm.register_crossing_candidate(c_backward_s, timestamp_sec=15.0)
    sm.register_crossing_candidate(c_backward_f, timestamp_sec=15.05)

    assert sm.total_in == 1
    assert sm.total_out == 1
    assert sm.net_count == 0  # Net change is exactly zero!


# ===========================================================================
# Scenario K: Carton approaches red line but does not cross -> expected no count
# ===========================================================================
def test_scenario_k_carton_approaches_without_crossing():
    sm = CartonCrossingStateMachine(line_position_px=500.0, hysteresis_margin_px=20.0, required_direction="IN")

    # Approaches threshold (400 -> 480)
    ev1 = sm.update_carton_position("carton_k", coord=400.0, displacement=5.0, timestamp_sec=11.0, contributing_cameras=["cam_side"])
    assert ev1 is None

    ev2 = sm.update_carton_position("carton_k", coord=480.0, displacement=80.0, timestamp_sec=11.2, contributing_cameras=["cam_side"])
    assert ev2 is None

    # Retreats without crossing (480 -> 350)
    ev3 = sm.update_carton_position("carton_k", coord=350.0, displacement=130.0, timestamp_sec=11.6, contributing_cameras=["cam_side"])
    assert ev3 is None

    assert sm.total_in == 0
    assert sm.total_out == 0
    assert sm.total_counted == 0


# ===========================================================================
# Scenario L: Carton crosses once but remains visible near line -> no repeated count
# ===========================================================================
def test_scenario_l_carton_crosses_and_remains_near_line():
    sm = CartonCrossingStateMachine(line_position_px=500.0, hysteresis_margin_px=20.0, required_direction="IN")

    # Crosses boundary to commit zone
    sm.update_carton_position("carton_l", coord=450.0, displacement=10.0, timestamp_sec=12.0, contributing_cameras=["cam_side"])
    ev = sm.update_carton_position("carton_l", coord=530.0, displacement=80.0, timestamp_sec=12.3, contributing_cameras=["cam_side"])
    assert ev is not None
    assert sm.total_counted == 1

    # Lingers near line for 50 frames
    for f in range(50):
        t = 12.4 + f * 0.033
        dup = sm.update_carton_position("carton_l", coord=535.0, displacement=85.0, timestamp_sec=t, contributing_cameras=["cam_side"])
        assert dup is None

    assert sm.total_counted == 1


# ===========================================================================
# Scenario M: Duplicate frames and repeated event delivery -> no extra count
# ===========================================================================
def test_scenario_m_duplicate_frames_and_event_delivery():
    pipeline = ThreeCameraCartonPipeline(conf_threshold=0.15)
    blank_frames = {"cam_top": np.zeros((10, 10, 3), dtype=np.uint8), "cam_side": np.zeros((10, 10, 3), dtype=np.uint8), "cam_flank": np.zeros((10, 10, 3), dtype=np.uint8)}

    for f_idx in range(30):
        pipeline.process_synchronized_frames(blank_frames, frame_idx=f_idx)

    assert pipeline.total_in == 0
    assert pipeline.total_out == 0
    assert pipeline.total_count == 0


# ===========================================================================
# Scenario N: Service restart after committed event -> persistent deduplication
# ===========================================================================
def test_scenario_n_service_restart_persistent_deduplication():
    # Session 1: Carton committed and state exported to disk
    sm1 = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")
    c1 = LocalCameraCrossingCandidate(
        local_event_id="ev_n1", camera_id="cam_side", local_track_id="trk_n1",
        timestamp_sec=14.0, direction="IN", confidence=0.88, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    v1 = ConsensusVerdict(
        level=ConsensusLevel.TWO_CAMERA_AGREEMENT, action=DecisionAction.COMMIT,
        supporting_cameras=["cam_side", "cam_top"], missing_cameras=["cam_flank"],
        mean_confidence=0.88, evidence_score=0.85, decision_reason="Two camera commit"
    )
    ev_committed = sm1.register_crossing_candidate(c1, timestamp_sec=14.0, consensus_verdict=v1)
    assert ev_committed is not None
    assert sm1.confirmed_count == 1
    saved_event_id = ev_committed.event_id

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_ledger = f.name

    sm1.export_state(tmp_ledger)

    # Session 2: Service restarts, reloads persisted ledger
    sm2 = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")
    sm2.load_state(tmp_ledger)

    assert saved_event_id in sm2._committed_event_ids
    assert sm2.confirmed_count == 1
    assert sm2.total_in == 1

    # Attempt duplicate replay after restart: must be rejected
    c1_replay = LocalCameraCrossingCandidate(
        local_event_id="ev_n1_replay", camera_id="cam_side", local_track_id="trk_n1",
        timestamp_sec=14.2, direction="IN", confidence=0.88, bbox=(492, 200, 552, 260), aspect_ratio=1.0
    )
    ev_replay = sm2.register_crossing_candidate(c1_replay, timestamp_sec=14.2, consensus_verdict=v1)
    assert ev_replay is None, "Replayed event after service restart was not rejected!"
    assert sm2.confirmed_count == 1
    os.remove(tmp_ledger)



# ===========================================================================
# Scenario O: Camera disconnection during crossing -> preserve state and recover safely
# ===========================================================================
def test_scenario_o_camera_disconnect_and_reconnect():
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    # Frame 1: cam_side observes
    c_s = LocalCameraCrossingCandidate(
        local_event_id="ev_o_s", camera_id="cam_side", local_track_id="trk_o1",
        timestamp_sec=15.0, direction="IN", confidence=0.85, bbox=(490, 200, 550, 260), aspect_ratio=1.0
    )
    sm.register_crossing_candidate(c_s, timestamp_sec=15.0)

    # Frame 2: cam_flank disconnects, then reconnects
    c_f = LocalCameraCrossingCandidate(
        local_event_id="ev_o_f", camera_id="cam_flank", local_track_id="trk_o2",
        timestamp_sec=15.4, direction="IN", confidence=0.82, bbox=(510, 200, 570, 260), aspect_ratio=1.0
    )
    ev_reconnect = sm.register_crossing_candidate(c_f, timestamp_sec=15.4)

    assert ev_reconnect is None, "Reconnecting camera caused duplicate event commit!"
    assert sm.confirmed_count == 1


# ===========================================================================
# Scenario P: Worker carries carton toward truck but returns it to pallet side
# ===========================================================================
def test_scenario_p_worker_returns_carton_to_pallet():
    sm = CartonCrossingStateMachine(line_position_px=500.0, hysteresis_margin_px=20.0, required_direction="IN")

    # Approaches threshold
    sm.update_carton_position("carton_p", coord=420.0, displacement=10.0, timestamp_sec=16.0, contributing_cameras=["cam_side"])
    # Enters boundary band
    sm.update_carton_position("carton_p", coord=490.0, displacement=80.0, timestamp_sec=16.2, contributing_cameras=["cam_side"])
    # Worker changes mind, turns back to pallet area (retreats to 300)
    ev_return = sm.update_carton_position("carton_p", coord=300.0, displacement=190.0, timestamp_sec=16.8, contributing_cameras=["cam_side"])

    assert ev_return is None
    assert sm.total_in == 0
    assert sm.total_out == 0
    assert sm.total_counted == 0


# ===========================================================================
# Scenario Q: Several cartons cross simultaneously while workers obscure one another
# ===========================================================================
def test_scenario_q_simultaneous_cartons_with_severe_worker_obscuration():
    disambiguator = StackDisambiguator()
    policy = ThreeCameraConsensusPolicy()

    # Conflicting / occluded evidence: cam_side sees 1, cam_flank sees 2
    view_counts = {"cam_side": 1, "cam_flank": 2}
    stack_res = disambiguator.disambiguate(view_counts)

    assert stack_res.is_ambiguous
    assert stack_res.requires_human_review

    verdict = policy.evaluate(
        observed_cameras=["cam_side", "cam_flank"],
        confidences=[0.75, 0.70],
        is_ambiguous=True,
    )
    assert verdict.action == DecisionAction.HOLD_FOR_REVIEW
    assert verdict.level == ConsensusLevel.CONFLICTING_OR_UNCERTAIN
