"""
Unit and Integration Tests for Week 4: Stack Disambiguation (Volumetric Formula).

Covers:
1. Volumetric Formula:
   - N_top = footprint columns x rows from cam_top
   - H_side = vertical layers from cam_side (along truck depth)
   - H_flank = vertical layers from cam_flank (breaks perspective)
   - Total = sum over columns c of H_layer(c)
2. Decision Rules:
   - N_top == sum H_side == sum H_flank -> COMMIT (exact count)
   - Conflict within 1 carton -> HOLD FOR REVIEW (save 3s synchronized clip)
   - Irreconcilable (> 1 carton diff) -> AMBIGUOUS_STACK -> human 1-click audit
3. End-to-end CorridorStateMachine integration with StackDisambiguator
4. Human 1-click audit API (/api/3angle/audit_event)
"""

from __future__ import annotations

import pytest
from typing import Tuple

from apps.truck_loading_detection.three_camera_system.stack_disambiguator import (
    StackDisambiguator,
    StackDisambiguationResult,
    StackResolutionState,
)
from apps.truck_loading_detection.three_camera_system.corridor import (
    BundleState,
    CorridorStateMachine,
    TransitBundle,
)
from apps.truck_loading_detection.three_camera_system.consensus_policy import (
    ThreeCameraConsensusPolicy,
)
from apps.truck_loading_detection.app import app


# ===========================================================================
# 1. Exact Agreement -> COMMIT (exact count)
# ===========================================================================
def test_week4_exact_agreement_commit_single_carton():
    """N_top == 1, H_side == 1, H_flank == 1 -> COMMIT (exact count 1)."""
    dis = StackDisambiguator()
    res = dis.disambiguate({"cam_top": 1, "cam_side": 1, "cam_flank": 1})

    assert res.decision_action == "COMMIT"
    assert res.resolved_count == 1
    assert res.resolution_state == StackResolutionState.INDIVIDUALLY_RESOLVED
    assert res.is_ambiguous is False
    assert res.requires_human_review is False
    assert res.audit_recommended is False
    assert res.save_clip_sec == 0.0


def test_week4_exact_agreement_commit_pure_vertical_stack():
    """Pure vertical stack of 3 cartons verified by both side perspectives."""
    dis = StackDisambiguator()
    res = dis.disambiguate({"cam_top": 1, "cam_side": 3, "cam_flank": 3})

    assert res.decision_action == "COMMIT"
    assert res.resolved_count == 3
    assert res.resolution_state == StackResolutionState.INDIVIDUALLY_RESOLVED
    assert res.is_ambiguous is False
    assert res.requires_human_review is False
    assert res.formula_breakdown["N_top"] == 1
    assert res.formula_breakdown["H_side"] == 3
    assert res.formula_breakdown["H_flank"] == 3


def test_week4_exact_agreement_commit_3d_grid_stack():
    """
    3D Grid Stack: 2 columns footprint (top view) x 3 vertical layers (side/flank views)
    Total = 2 x 3 = 6 cartons.
    """
    dis = StackDisambiguator()
    res = dis.disambiguate({"cam_top": 2, "cam_side": 3, "cam_flank": 3})

    assert res.decision_action == "COMMIT"
    assert res.resolved_count == 6
    assert res.resolution_state == StackResolutionState.INDIVIDUALLY_RESOLVED
    assert res.is_ambiguous is False
    assert res.requires_human_review is False


def test_week4_exact_agreement_non_uniform_columns():
    """Non-uniform columns: Total = sum over columns c of H_layer(c)."""
    dis = StackDisambiguator()
    # 2 columns with heights 2 and 3 -> Total = 5
    res = dis.disambiguate(
        {"cam_top": 2, "cam_side": 3, "cam_flank": 3},
        column_layer_heights=[2, 3],
    )
    assert res.decision_action == "COMMIT"
    assert res.resolved_count == 5


# ===========================================================================
# 2. Conflict within 1 Carton -> HOLD FOR REVIEW (save 3s synchronized clip)
# ===========================================================================
def test_week4_conflict_within_one_carton_holds_for_review_3s_clip():
    """
    Conflict within 1 carton (e.g. side sees 2, flank sees 3, diff = 1):
    -> HOLD FOR REVIEW (save 3s synchronized clip).
    """
    dis = StackDisambiguator()
    res = dis.disambiguate({"cam_top": 1, "cam_side": 2, "cam_flank": 3})

    assert res.decision_action == "HOLD_FOR_REVIEW"
    assert res.resolved_count == 3
    assert res.resolution_state == StackResolutionState.AMBIGUOUS_STACK
    assert res.is_ambiguous is True
    assert res.requires_human_review is True
    assert res.save_clip_sec == 3.0
    assert res.audit_recommended is False
    assert "save 3s synchronized clip" in res.formula_breakdown["action"]


def test_week4_conflict_within_one_carton_side_1_flank_2():
    """Conflict within 1 carton (side=1, flank=2, diff=1) -> HOLD FOR REVIEW (3s clip)."""
    dis = StackDisambiguator()
    res = dis.disambiguate({"cam_side": 1, "cam_flank": 2})

    assert res.decision_action == "HOLD_FOR_REVIEW"
    assert res.resolved_count == 2
    assert res.save_clip_sec == 3.0
    assert res.is_ambiguous is True
    assert res.requires_human_review is True


# ===========================================================================
# 3. Irreconcilable (> 1 carton discrepancy) -> AMBIGUOUS_STACK -> human 1-click audit
# ===========================================================================
def test_week4_irreconcilable_conflict_triggers_human_audit():
    """
    Discrepancy > 1 carton (e.g. side sees 4, flank sees 2, diff = 2):
    -> AMBIGUOUS_STACK -> human 1-click audit.
    """
    dis = StackDisambiguator()
    res = dis.disambiguate({"cam_top": 1, "cam_side": 4, "cam_flank": 2})

    assert res.decision_action == "AMBIGUOUS_STACK"
    assert res.resolution_state == StackResolutionState.AMBIGUOUS_STACK
    assert res.is_ambiguous is True
    assert res.requires_human_review is True
    assert res.audit_recommended is True
    assert res.save_clip_sec == 3.0
    assert "human 1-click audit" in res.formula_breakdown["action"]


def test_week4_irreconcilable_conflict_diff_3():
    """Discrepancy of 3 cartons (side=5, flank=2) -> AMBIGUOUS_STACK -> human 1-click audit."""
    dis = StackDisambiguator()
    res = dis.disambiguate({"cam_top": 1, "cam_side": 5, "cam_flank": 2})

    assert res.decision_action == "AMBIGUOUS_STACK"
    assert res.audit_recommended is True
    assert res.resolved_count == 5


# ===========================================================================
# 4. End-to-End Corridor State Machine Integration
# ===========================================================================
class MockTrk:
    def __init__(self, tid: str, center: Tuple[float, float], qty: int = 1):
        self.local_track_id = tid
        self.center = center
        self.current_bbox = (center[0]-30, center[1]-30, center[0]+30, center[1]+30)
        self.current_confidence = 0.88
        self.estimated_qty = qty
        self.worker_id = "worker_w4"


def test_corridor_state_machine_with_week4_stack_disambiguation():
    """
    Corridor state machine updates transit bundle and applies Week 4 StackDisambiguator
    to trigger exact COMMIT or HOLD_FOR_REVIEW with 3s clip metadata.
    """
    sm = CorridorStateMachine(coalesce_window_sec=6.0)
    dis = StackDisambiguator()
    consensus = ThreeCameraConsensusPolicy()

    # Worker carries stack of 2 cartons into dock threshold
    trk_side = MockTrk("side_01", (650.0, 300.0), qty=2)
    trk_flank = MockTrk("flank_01", (450.0, 300.0), qty=2)
    trk_top = MockTrk("top_01", (650.0, 300.0), qty=1)

    sm.update(
        {"cam_side": [trk_side], "cam_flank": [trk_flank], "cam_top": [trk_top]},
        timestamp_sec=1.0,
        disambiguator=dis,
        consensus_policy=consensus,
    )

    # Moves into transit at t = 1.6s
    trk_side.center = (500.0, 300.0)
    trk_flank.center = (550.0, 300.0)
    trk_top.center = (500.0, 300.0)
    sm.update(
        {"cam_side": [trk_side], "cam_flank": [trk_flank], "cam_top": [trk_top]},
        timestamp_sec=1.6,
        disambiguator=dis,
        consensus_policy=consensus,
    )

    # Fully exits corridor into truck interior at t = 2.4s
    trk_side.center = (380.0, 300.0)
    trk_flank.center = (680.0, 300.0)
    trk_top.center = (380.0, 300.0)
    commits = sm.update(
        {"cam_side": [trk_side], "cam_flank": [trk_flank], "cam_top": [trk_top]},
        timestamp_sec=2.4,
        disambiguator=dis,
        consensus_policy=consensus,
    )

    assert len(commits) == 1
    ev = commits[0]
    assert ev["carton_count"] == 2
    assert ev["decision_action"] == "COMMIT"
    assert ev["is_auto_confirmed"] is True
    assert ev["requires_review"] is False


# ===========================================================================
# 5. Human 1-Click Audit API Endpoint (/api/3angle/audit_event)
# ===========================================================================
def test_human_1_click_audit_api_approve():
    """Test 1-click audit 'approve' action on a review event."""
    app.config["TESTING"] = True
    client = app.test_client()

    # Test audit_event with mock event in active session
    with app.test_request_context():
        auth_headers = {"Authorization": "Bearer op_token_dock_1"}
        # Missing event_id -> 400
        res_bad = client.post("/api/3angle/audit_event", json={"action": "approve"},
                              headers=auth_headers)
        assert res_bad.status_code == 400

        # Seed mock review event into offline_jobs
        from apps.truck_loading_detection.app import offline_jobs
        test_job_id = "job_audit_test"
        mock_ev = {
            "event_id": "ev_audit_01",
            "global_carton_id": "gc_audit_01",
            "carton_count": 2,
            "fused_cartons": 2,
            "is_auto_confirmed": False,
            "requires_review": True,
            "state": "AMBIGUOUS_STACK",
            "decision_action": "AMBIGUOUS_STACK",
        }
        offline_jobs[test_job_id] = {
            "report": {
                "result": {
                    "events": [mock_ev],
                }
            }
        }

        # 1. Test 1-click 'adjust' to 3 cartons
        res_adj = client.post("/api/3angle/audit_event", json={
            "job_id": test_job_id,
            "event_id": "ev_audit_01",
            "action": "adjust",
            "adjusted_count": 3,
        }, headers=auth_headers)
        assert res_adj.status_code == 200
        data_adj = res_adj.get_json()
        assert data_adj["event"]["carton_count"] == 3
        # 'adjust' => MARK_AMBIGUOUS -> not auto-confirmed, not human-approved, review required
        assert data_adj["event"]["is_auto_confirmed"] is False
        assert data_adj["event"]["requires_review"] is True

        # 2. Test 1-click 'approve'
        res_app = client.post("/api/3angle/audit_event", json={
            "job_id": test_job_id,
            "event_id": "ev_audit_01",
            "action": "approve",
        }, headers=auth_headers)
        assert res_app.status_code == 200
        data_app = res_app.get_json()
        assert data_app["event"]["audit_verdict"] == "HUMAN_APPROVED"
        assert data_app["event"]["state"] == "COUNT_COMMITTED"

        # 3. Test 1-click 'reject'
        res_rej = client.post("/api/3angle/audit_event", json={
            "job_id": test_job_id,
            "event_id": "ev_audit_01",
            "action": "reject",
        }, headers=auth_headers)
        assert res_rej.status_code == 200
        data_rej = res_rej.get_json()
        assert data_rej["event"]["audit_verdict"] == "HUMAN_REJECTED"
        assert data_rej["event"]["action"] == "DISCARD"
