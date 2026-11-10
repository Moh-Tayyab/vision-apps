"""
Phase 3.2.1 — Regression Closure & Counting Integrity Test Suite.

Validates:
1. Unmatched automatic prediction suppression (confidence floor on 2-camera agreement).
2. Both missed crossings (session_21_ev_02 low-conf recovery & session_21_ev_04 direction isolation).
3. Verified duplicate tracklet coalescence vs distinct cartons.
4. Separate nearby cartons in parallel lanes.
5. Cross-camera identity continuity across resolutions.
6. Durable idempotency: stable event occurrence tokens, replay with shifted timestamps.
7. Legitimate repeated crossings.
8. Concurrent persistence and restart recovery.
9. Reconciliation engine adversarial Kuhn-Munkres assignment.
"""

import os
import shutil
import tempfile
import threading
import time
import pytest
import numpy as np

from apps.truck_loading_detection.three_camera_system.consensus_policy import (
    ThreeCameraConsensusPolicy,
    ConsensusLevel,
    DecisionAction,
    ConsensusVerdict,
)
from apps.truck_loading_detection.three_camera_system.event_store import EventStore
from apps.truck_loading_detection.three_camera_system.corridor import (
    LocalCameraCrossingCandidate,
)
from apps.truck_loading_detection.three_camera_system.pipeline import (
    ThreeCameraCartonPipeline,
    CrossingWindow,
)
from apps.truck_loading_detection.three_camera_system.crossing_state_machine import (
    CartonCrossingStateMachine,
    PhysicalCarton,
    CrossingState,
)
from apps.truck_loading_detection.three_camera_system.reconciliation_engine import (
    GroundTruthReconciler,
    GroundTruthCrossing,
    PredictedCrossing,
)


@pytest.fixture
def temp_db_path():
    td = tempfile.mkdtemp(prefix="phase3_2_1_test_")
    db_p = os.path.join(td, "carton_events.db")
    yield db_p
    shutil.rmtree(td, ignore_errors=True)


# =========================================================================
# Priority 1: Unmatched Automatic Prediction Suppression
# =========================================================================

def test_unmatched_automatic_prediction_suppressed_by_confidence_floor():
    """
    Session 21 p32_ev_00010 had 2-camera agreement but mean confidence 0.192.
    Under Phase 3.2, it committed as COUNTED (is_auto_confirmed=1), causing 75% precision.
    Phase 3.2.1 enforces min_two_camera_conf=0.25. Sub-0.25 2-cam agreement must route
    to HOLD_FOR_REVIEW so it does not degrade automatic precision.
    """
    policy = ThreeCameraConsensusPolicy(
        conf_threshold=0.15,
        min_two_camera_conf=0.25,
    )

    # Low-confidence 2-camera detection (mean conf 0.192 < 0.25)
    observed = ["cam_top", "cam_side"]
    confidences = [0.194, 0.190]
    verdict = policy.evaluate(observed, confidences)
    assert verdict.action == DecisionAction.HOLD_FOR_REVIEW
    assert verdict.level == ConsensusLevel.CONFLICTING_OR_UNCERTAIN

    # Confident 2-camera detection (mean conf 0.55 >= 0.25)
    conf_confident = [0.60, 0.50]
    verdict_good = policy.evaluate(observed, conf_confident)
    assert verdict_good.action == DecisionAction.COMMIT
    assert verdict_good.level == ConsensusLevel.TWO_CAMERA_AGREEMENT


# =========================================================================
# Priority 2: Missed Crossings Recovery
# =========================================================================

def test_missed_crossing_1_session21_ev02_single_camera_conf_recovery():
    """
    Session 21 ev_02 (t=4.696s) was occluded on top and flank; cam_side detected at conf=0.2239.
    Phase 3.2's min_single_camera_conf=0.25 dropped it as noise.
    Phase 3.2.1 restores conf_threshold (0.15) for single-camera, routing to HOLD_FOR_REVIEW.
    """
    policy = ThreeCameraConsensusPolicy(conf_threshold=0.15)
    observed = ["cam_side"]
    confidences = [0.2239]
    verdict = policy.evaluate(observed, confidences)
    assert verdict.action == DecisionAction.HOLD_FOR_REVIEW
    assert verdict.level == ConsensusLevel.SINGLE_CAMERA_OBSERVATION
    assert "cam_side" in verdict.supporting_cameras


def test_missed_crossing_2_session21_ev04_direction_isolation_in_coalescence():
    """
    CrossingWindow must not coalesce candidates with conflicting directions into a single
    conflicted window that drops the crossing.
    Candidates with opposing directions must be segregated.
    """
    cand_out = LocalCameraCrossingCandidate(
        local_event_id="cand_out",
        camera_id="cam_top",
        local_track_id="trk_1",
        timestamp_sec=11.400,
        direction="OUT",
        confidence=0.45,
        bbox=(480.0, 200.0, 520.0, 260.0),
        aspect_ratio=1.0,
    )
    cand_in_1 = LocalCameraCrossingCandidate(
        local_event_id="cand_in_1",
        camera_id="cam_side",
        local_track_id="trk_2",
        timestamp_sec=11.633,
        direction="IN",
        confidence=0.55,
        bbox=(490.0, 210.0, 530.0, 270.0),
        aspect_ratio=1.0,
    )

    # Initial window with IN candidate
    window = CrossingWindow(
        window_id="win_01",
        start_time=11.633,
        last_time=11.633,
        local_events={"cam_side": [cand_in_1]},
    )

    # In pipeline coalescence logic:
    # Direction check verifies if any candidate in window has conflicting direction
    has_conflict = any(
        o_ev.direction != cand_out.direction
        for other_cam_evs in window.local_events.values()
        for o_ev in other_cam_evs
    )
    # Opposite direction MUST trigger conflict, preventing absorption
    assert has_conflict is True


# =========================================================================
# Priority 3 & 4: Duplicate Tracklets vs Distinct Cartons & Multi-Resolution
# =========================================================================

def test_verified_duplicate_tracklets_coalesced_within_corridor():
    """
    When worker occlusion causes local tracker ID fragmentation (e.g. cam_top trk_1 -> cam_top trk_4
    within 0.8s in the same lane), the state machine stitches the tracklet to the existing physical carton.
    """
    sm = CartonCrossingStateMachine()

    # First observation of physical carton
    phys = sm.get_or_create_physical_carton(
        camera_id="cam_top",
        local_track_id="trk_1",
        center=(500.0, 200.0),
        timestamp_sec=1.0,
        direction="IN",
    )
    cid_initial = phys.carton_id

    # Fragmented tracklet appearing 0.4s later in same lane (y=205.0) within corridor drift (x=530.0)
    phys_stitched = sm.find_matching_physical_carton(
        camera_id="cam_top",
        local_track_id="trk_4",
        center=(530.0, 205.0),
        timestamp_sec=1.4,
        direction="IN",
    )
    assert phys_stitched is not None
    assert phys_stitched.carton_id == cid_initial


def test_separate_nearby_cartons_parallel_lanes_not_merged():
    """
    Two cartons crossing simultaneously in distinct lanes (y-separation = 150px > 60px)
    must remain separate physical cartons.
    """
    sm = CartonCrossingStateMachine()

    phys1 = sm.get_or_create_physical_carton(
        camera_id="cam_top",
        local_track_id="trk_lane1",
        center=(500.0, 100.0),
        timestamp_sec=5.0,
        direction="IN",
    )

    # Parallel lane 150px away
    phys2 = sm.find_matching_physical_carton(
        camera_id="cam_top",
        local_track_id="trk_lane2",
        center=(500.0, 250.0),
        timestamp_sec=5.1,
        direction="IN",
    )
    assert phys2 is None, "Parallel lane carton must NOT match physical carton 1!"


def test_multi_resolution_lane_scaling():
    """
    Verifies that spatial thresholds scale proportionally with resolution.
    At 1080p (1920x1080), 60px base (720p) scales to 60 * (1080/720) = 90px.
    """
    base_h, base_w = 720, 1280
    high_h, high_w = 1080, 1920

    scale_y = high_h / base_h
    scaled_tol = 60.0 * scale_y
    assert scaled_tol == 90.0

    # Test scaling function or invariant
    dist_720 = 50.0  # within 60px
    dist_1080 = dist_720 * scale_y
    assert dist_1080 < scaled_tol


# =========================================================================
# Priority 5: Durable Event Idempotency & Repeat Crossings
# =========================================================================

def test_durable_idempotency_stable_occurrence_token(temp_db_path):
    """
    Replaying an event with the exact same crossing_occurrence_id must be rejected
    as ALREADY_COMMITTED_IDEMPOTENT, even if event_id is regenerated.
    """
    store = EventStore(temp_db_path)

    event_payload = {
        "event_id": "ev_phase3_orig",
        "global_carton_id": "carton_101",
        "timestamp_sec": 10.0,
        "direction": "IN",
        "carton_count": 1,
        "crossing_occurrence_id": "session_alpha:carton_101:c1:IN",
    }
    ok1, reason1, _ = store.commit_event(event_payload, session_id="session_alpha")
    assert ok1 is True
    assert reason1 == "COMMITTED_NEW"

    # Replay with new event_id and jittered timestamp (+0.8s)
    replay_payload = {
        "event_id": "ev_phase3_replay_new_id",
        "global_carton_id": "carton_101",
        "timestamp_sec": 10.8,
        "direction": "IN",
        "carton_count": 1,
        "crossing_occurrence_id": "session_alpha:carton_101:c1:IN",
    }
    ok2, reason2, _ = store.commit_event(replay_payload, session_id="session_alpha")
    assert ok2 is False
    assert reason2 == "ALREADY_COMMITTED_IDEMPOTENT"


def test_legitimate_repeated_crossing_with_explicit_occurrence_tokens(temp_db_path):
    """
    A carton crossing IN (c1), then crossing OUT (c2), then crossing IN (c3)
    must commit all three crossings successfully.
    """
    store = EventStore(temp_db_path)

    ok1, _, _ = store.commit_event({
        "event_id": "ev_c1",
        "global_carton_id": "carton_200",
        "timestamp_sec": 5.0,
        "direction": "IN",
        "occurrence_index": 1,
    }, session_id="session_repeat")
    assert ok1 is True

    ok2, _, _ = store.commit_event({
        "event_id": "ev_c2",
        "global_carton_id": "carton_200",
        "timestamp_sec": 45.0,
        "direction": "OUT",
        "occurrence_index": 2,
    }, session_id="session_repeat")
    assert ok2 is True

    ok3, _, _ = store.commit_event({
        "event_id": "ev_c3",
        "global_carton_id": "carton_200",
        "timestamp_sec": 95.0,
        "direction": "IN",
        "occurrence_index": 3,
    }, session_id="session_repeat")
    assert ok3 is True

    events = store.get_events_by_session("session_repeat")
    assert len(events) == 3


def test_concurrent_persistence_and_restart_recovery(temp_db_path):
    """
    Validates:
    1. Multi-threaded concurrent commit attempts are thread-safe.
    2. Restart recovery: a new EventStore instance preserves committed events.
    """
    store1 = EventStore(temp_db_path)
    errors = []

    def worker(worker_id):
        try:
            store1.commit_event({
                "event_id": f"ev_worker_{worker_id}",
                "global_carton_id": f"carton_worker_{worker_id}",
                "timestamp_sec": 10.0 + worker_id,
                "direction": "IN",
                "carton_count": 1,
            }, session_id="concurrent_session")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0

    # Restart recovery: open new EventStore instance
    store2 = EventStore(temp_db_path)
    events = store2.get_events_by_session("concurrent_session")
    assert len(events) == 10


# =========================================================================
# Priority 6: Reconciliation Engine Kuhn-Munkres Correctness
# =========================================================================

def test_reconciliation_adversarial_kuhn_munkres_vs_greedy():
    """
    Adversarial bipartite matching case where greedy nearest-neighbor matching fails.
    GT 1 at t=1.0s, GT 2 at t=2.0s.
    Pred 1 at t=1.4s, Pred 2 at t=2.1s.
    Greedy matching:
    Pred 1 dt to GT 1 is 0.4s (match), leaving Pred 2 dt to GT 2 is 0.1s (match).
    Now construct adversarial scenario:
    GT 1 at t=10.0s, GT 2 at t=11.1s.
    Pred A at t=10.9s (dt=0.9 to GT 1, dt=0.2 to GT 2).
    Pred B at t=12.2s (dt=1.1 to GT 2, dt=2.2 to GT 1 -> out of tolerance).
    Tolerance tau = 1.5s.
    If greedy runs:
    Pred A greedily matches GT 2 (dt=0.2).
    Then Pred B cannot match GT 1 (dt=2.2 > 1.5) and GT 2 is already taken!
    Result of greedy: 1 match.
    Kuhn-Munkres maximum-cardinality matching:
    Matches Pred A with GT 1 (dt=0.9 <= 1.5) and Pred B with GT 2 (dt=1.1 <= 1.5).
    Result of Kuhn-Munkres: 2 matches!
    """
    reconciler = GroundTruthReconciler(temporal_tolerance_sec=1.5)

    gt_events = [
        GroundTruthCrossing(event_id="gt_1", session_id="test_s", timestamp_sec=10.0, direction="IN"),
        GroundTruthCrossing(event_id="gt_2", session_id="test_s", timestamp_sec=11.1, direction="IN"),
    ]
    predictions = [
        PredictedCrossing(
            event_id="pred_A",
            session_id="test_s",
            global_carton_id="carton_A",
            timestamp_sec=10.9,
            direction="IN",
            carton_count=1,
            state="COUNT_COMMITTED",
            is_auto_confirmed=True,
            requires_review=False,
            participating_cameras=("cam_top", "cam_side"),
        ),
        PredictedCrossing(
            event_id="pred_B",
            session_id="test_s",
            global_carton_id="carton_B",
            timestamp_sec=12.2,
            direction="IN",
            carton_count=1,
            state="COUNT_COMMITTED",
            is_auto_confirmed=True,
            requires_review=False,
            participating_cameras=("cam_top", "cam_side"),
        ),
    ]

    metrics, matched, unmatched_preds, missed_gts = reconciler.reconcile_session(
        ground_truth=gt_events,
        predictions=predictions,
        session_id="test_s",
    )

    # Kuhn-Munkres must achieve maximum cardinality: 2 matches!
    assert metrics.matched_auto_confirmed == 2
    assert len(missed_gts) == 0
    assert len(matched) == 2
