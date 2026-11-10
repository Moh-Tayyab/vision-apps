"""
tests/test_phase3_3_operational_accuracy.py

Comprehensive regression test suite for Phase 3.3:
- Review approval and rejection
- Review audit history
- Repeated operator actions
- Count idempotency
- Shadow-mode inventory isolation
- Missing annotation manifests
- Occlusion evidence handling
- Confidence calibration reproducibility
- Locked-test-set protection
- Restart recovery
- Multi-camera evidence consistency
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_repo_root = Path(__file__).resolve().parent.parent
_app_dir = _repo_root / "apps" / "truck_loading_detection"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app import app
from three_camera_system.annotation_schema import (
    GroundTruthEventAnnotation,
    MultiSessionDatasetManager,
    SessionAnnotationRecord,
    VideoProvenanceRecord,
)
from three_camera_system.consensus_policy import (
    ConsensusLevel,
    DecisionAction,
    ThreeCameraConsensusPolicy,
)
from three_camera_system.event_store import EventStore
from three_camera_system.shadow_mode import ShadowModeManager, ShadowStreamHealth


@pytest.fixture
def temp_event_store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name
    store = EventStore(db_path=db_path)
    yield store
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# 1. Review Approval and Rejection
def test_review_approval_and_rejection(temp_event_store):
    store = temp_event_store
    session_id = "test_review_sess"

    # Commit event requiring review
    ev_data = {
        "event_id": "ev_rev_001",
        "global_carton_id": "carton_001",
        "state": "REVIEW_REQUIRED",
        "is_auto_confirmed": 0,
        "requires_review": 1,
        "direction": "IN",
        "carton_count": 1,
        "participating_cameras": ["cam_side"],
    }
    store.commit_event(ev_data, session_id=session_id)

    # Approve
    ok, reason, updated = store.record_manual_review(
        event_id="ev_rev_001",
        action="CONFIRM_CARTON_CROSSING",
        operator_id="operator_01",
        reason="Verified clear carton on side camera",
    )
    assert ok is True
    assert updated["state"] == "MANUALLY_CONFIRMED"
    assert updated["is_human_approved"] is True
    assert updated["is_auto_confirmed"] is False  # CRITICAL: Partitioned from auto
    assert updated["requires_review"] is False

    # Commit second event and Reject
    ev_data_2 = {
        "event_id": "ev_rev_002",
        "global_carton_id": "carton_002",
        "state": "REVIEW_REQUIRED",
        "is_auto_confirmed": 0,
        "requires_review": 1,
        "direction": "IN",
        "carton_count": 1,
        "participating_cameras": ["cam_flank"],
    }
    store.commit_event(ev_data_2, session_id=session_id)

    ok2, reason2, updated2 = store.record_manual_review(
        event_id="ev_rev_002",
        action="REJECT_FALSE_DETECTION",
        operator_id="operator_01",
        reason="Forklift shadow artifact",
    )
    assert ok2 is True
    assert updated2["state"] == "MANUALLY_REJECTED"
    assert updated2["is_human_approved"] is False
    assert updated2["is_auto_confirmed"] is False
    assert updated2["requires_review"] is False


# 2. Review Audit History
def test_review_audit_history(temp_event_store):
    store = temp_event_store
    session_id = "audit_hist_sess"

    ev_data = {
        "event_id": "ev_audit_01",
        "global_carton_id": "carton_audit_01",
        "state": "REVIEW_REQUIRED",
        "is_auto_confirmed": 0,
        "requires_review": 1,
        "direction": "IN",
        "carton_count": 1,
    }
    store.commit_event(ev_data, session_id=session_id)

    store.record_manual_review(
        event_id="ev_audit_01",
        action="MARK_AMBIGUOUS",
        operator_id="operator_dock_A",
        reason="Extreme glare on side camera",
    )
    store.record_manual_review(
        event_id="ev_audit_01",
        action="CONFIRM_CARTON_CROSSING",
        operator_id="supervisor_B",
        reason="Confirmed with overhead secondary view",
    )

    audit_logs = store.get_review_audit_log(event_id="ev_audit_01")
    assert len(audit_logs) == 2
    assert audit_logs[0]["action"] == "MARK_AMBIGUOUS"
    assert audit_logs[0]["operator_id"] == "operator_dock_A"
    assert audit_logs[1]["action"] == "CONFIRM_CARTON_CROSSING"
    assert audit_logs[1]["operator_id"] == "supervisor_B"


# 3. Repeated Operator Actions & Count Idempotency
def test_repeated_operator_actions_and_count_idempotency(temp_event_store):
    store = temp_event_store
    session_id = "idempotency_sess"

    ev_data = {
        "event_id": "ev_idem_01",
        "global_carton_id": "carton_idem_01",
        "state": "REVIEW_REQUIRED",
        "is_auto_confirmed": 0,
        "requires_review": 1,
        "direction": "IN",
        "carton_count": 1,
    }
    store.commit_event(ev_data, session_id=session_id)

    # First approval
    ok1, reason1, _ = store.record_manual_review(
        event_id="ev_idem_01",
        action="CONFIRM_CARTON_CROSSING",
        operator_id="op_clicker",
        reason="First approval click",
    )
    assert ok1 is True
    assert reason1 == "ACTION_APPLIED"

    # Second approval click (rapid multi-click simulation)
    ok2, reason2, _ = store.record_manual_review(
        event_id="ev_idem_01",
        action="CONFIRM_CARTON_CROSSING",
        operator_id="op_clicker",
        reason="Repeated click attempt",
    )
    assert ok2 is True
    assert reason2 == "ALREADY_CONFIRMED_IDEMPOTENT"

    # Verify inventory count is strictly 1, NEVER 2
    counts = store.get_counts(session_id=session_id)
    assert counts["human_approved_count"] == 1
    assert counts["confirmed_count"] == 1
    assert counts["auto_confirmed_count"] == 0


# 4. Shadow Mode Inventory Isolation
def test_shadow_mode_inventory_isolation(temp_event_store):
    store = temp_event_store
    shadow = ShadowModeManager(
        event_store=store,
        session_id="shadow_audit_01",
        authoritative_inventory_locked=True,
    )
    assert shadow.authoritative_inventory_locked is True

    # Register physical ground truth
    shadow.set_human_physical_count(10)

    # Commit 2 auto-confirmed events and 1 review event
    store.commit_event({
        "event_id": "sh_ev_1",
        "global_carton_id": "cid_sh_1",
        "direction": "IN",
        "carton_count": 1,
        "state": "COUNT_COMMITTED",
        "is_auto_confirmed": 1,
    }, session_id="shadow_audit_01")
    store.commit_event({
        "event_id": "sh_ev_2",
        "global_carton_id": "cid_sh_2",
        "direction": "IN",
        "carton_count": 1,
        "state": "COUNT_COMMITTED",
        "is_auto_confirmed": 1,
    }, session_id="shadow_audit_01")
    store.commit_event({
        "event_id": "sh_ev_3",
        "global_carton_id": "cid_sh_3",
        "direction": "IN",
        "carton_count": 1,
        "state": "REVIEW_REQUIRED",
        "is_auto_confirmed": 0,
        "requires_review": 1,
    }, session_id="shadow_audit_01")

    # Operator approves review event
    store.record_manual_review("sh_ev_3", "CONFIRM_CARTON_CROSSING", "op_shadow", "Approved")

    summary = shadow.get_shadow_counts()
    assert summary["authoritative_inventory_locked"] is True
    assert summary["human_physical_count"] == 10
    assert summary["automatic_confirmed_count"] == 2
    assert summary["human_approved_count"] == 1
    assert summary["system_total_confirmed"] == 3
    assert summary["discrepancy_count"] == 7  # 10 - 3


# 5. Missing Annotation Manifests
def test_missing_annotation_manifests():
    manifest_path = _repo_root / "data/ground_truth/dataset_manifest_v3_3.json"
    assert manifest_path.exists()
    manager = MultiSessionDatasetManager(str(manifest_path))

    # Verify unannotated sessions 1-3 are flagged as MANIFEST_MISSING with documented blocker
    assert "aligned_session_01" in manager.sessions
    s1 = manager.sessions["aligned_session_01"]
    assert s1.status.value == "MANIFEST_MISSING"
    assert s1.blocker_reason is not None
    assert len(s1.events) == 0  # CRITICAL: Never invent timestamps or labels

    missing_sessions = [s for s in manager.sessions.values() if s.status.value == "MANIFEST_MISSING"]
    assert len(missing_sessions) >= 3


# 6. Occlusion Evidence Handling
def test_occlusion_evidence_handling():
    policy = ThreeCameraConsensusPolicy(min_single_camera_conf=0.15, min_two_camera_conf=0.25)

    # 1. Single camera observation during worker chest occlusion
    verdict_single = policy.evaluate(
        observed_cameras=["cam_side"],
        confidences=[0.45],
    )
    # Must HOLD_FOR_REVIEW, NEVER commit autonomously
    assert verdict_single.action == DecisionAction.HOLD_FOR_REVIEW
    assert verdict_single.level == ConsensusLevel.SINGLE_CAMERA_OBSERVATION

    # 2. Worker presence alone without two-camera consensus must not commit
    # Even if conf is high on single camera
    verdict_single_high = policy.evaluate(
        observed_cameras=["cam_side"],
        confidences=[0.85],
    )
    assert verdict_single_high.action == DecisionAction.HOLD_FOR_REVIEW

    # 3. Two cameras agreeing commits
    verdict_two = policy.evaluate(
        observed_cameras=["cam_top", "cam_side"],
        confidences=[0.35, 0.40],
    )
    assert verdict_two.action == DecisionAction.COMMIT
    assert verdict_two.level == ConsensusLevel.TWO_CAMERA_AGREEMENT


# 7. Confidence Calibration Reproducibility
def test_confidence_calibration_reproducibility():
    calib_json = _repo_root / "docs/phase3/confidence_calibration_metrics.json"
    assert calib_json.exists()
    with open(calib_json, "r") as f:
        data = json.load(f)

    assert data["active_certified_configuration"]["min_single_camera_conf"] == 0.15
    assert data["active_certified_configuration"]["min_two_camera_mean_conf"] == 0.25
    assert data["rollback_configuration"]["min_single_camera_conf"] == 0.25
    assert data["rollback_configuration"]["min_two_camera_mean_conf"] == 0.30


# 8. Locked Test Set Protection
def test_locked_test_set_protection():
    calib_json = _repo_root / "docs/phase3/confidence_calibration_metrics.json"
    with open(calib_json, "r") as f:
        data = json.load(f)

    lock_info = data["locked_test_protection"]
    assert lock_info["locked_test_excluded"] is True
    assert "session_21" in lock_info["locked_test_sessions"]
    assert lock_info["locked_test_hash"] == "6d4b773e5a1a3ffaf39fb68269f209e2abb3ca1b0619b6065722333aa582d033"


# 9. Restart Recovery
def test_restart_recovery(temp_event_store):
    store = temp_event_store
    shadow = ShadowModeManager(event_store=store, session_id="restart_sess")
    shadow.set_human_physical_count(15)
    shadow.log_incident("SYNC_JITTER_EXCEEDED", "WARNING", "Drift 35ms")

    # Simulate restart by creating new ShadowModeManager on same store
    shadow_recovered = ShadowModeManager(event_store=store, session_id="restart_sess")
    summary = shadow_recovered.recover_session_state()

    assert shadow_recovered.human_physical_count == 15
    assert len(shadow_recovered.incidents) >= 2  # prior incident + RESTART_RECOVERY
    types = [inc.incident_type for inc in shadow_recovered.incidents]
    assert "RESTART_RECOVERY" in types


# 10. Multi-Camera Evidence Consistency
def test_multicamera_evidence_consistency():
    policy = ThreeCameraConsensusPolicy()

    # Unanimous
    v3 = policy.evaluate(
        observed_cameras=["cam_top", "cam_side", "cam_flank"],
        confidences=[0.40, 0.50, 0.60],
    )
    assert v3.action == DecisionAction.COMMIT
    assert v3.level == ConsensusLevel.THREE_CAMERA_UNANIMOUS

    # Direction conflict
    v_dir = policy.evaluate(
        observed_cameras=["cam_top", "cam_side"],
        confidences=[0.50, 0.50],
        has_direction_conflict=True,
    )
    assert v_dir.action == DecisionAction.HOLD_FOR_REVIEW
    assert v_dir.level == ConsensusLevel.CONFLICTING_OR_UNCERTAIN
