"""
Unit and Integration Tests for Week 5: Evaluate & Deploy.

Specification Requirements:
1. Metric: Carton-level exact match per event (CLEM), not mAP.
2. Target: >=95% exact match on visible cartons; AMBIGUOUS_STACK rate <5%.
3. Review queue on dashboard for flagged events; pallet scale as ground-truth cross-check.
"""

from __future__ import annotations

import csv
import json
import os
import pytest
from typing import Any, Dict, List

from apps.truck_loading_detection.three_camera_system.evaluator import (
    CartonLevelEvaluator,
    EventMatchResult,
    PalletScaleAuditResult,
    PalletScaleCrossChecker,
    PalletScaleReading,
    Week5EvaluationReport,
)
from apps.truck_loading_detection.three_camera_system.pipeline import (
    ThreeCameraCartonPipeline,
)
from apps.truck_loading_detection.app import app, state


# ============================================================================
# 1. Pallet Scale Ground-Truth Cross-Check Model Tests
# ============================================================================

def test_pallet_scale_reading_exact_tare_and_net_weight():
    """
    Verifies N_pallet = round((W_loaded - W_tare) / w_carton).
    Gross = 100.0 kg, Tare = 25.0 kg, Unit = 12.5 kg -> Net = 75.0 kg -> 6 cartons.
    """
    reading = PalletScaleReading(
        gross_weight_kg=100.0,
        tare_weight_kg=25.0,
        nominal_carton_weight_kg=12.5,
        weight_tolerance_pct=3.0,
    )
    assert reading.net_weight_kg == 75.0
    assert reading.estimated_cartons == 6
    w_min, w_max = reading.expected_weight_range
    # 75 kg +/- 3% (2.25 kg) -> [72.75, 77.25]
    assert pytest.approx(w_min, 0.01) == 72.75
    assert pytest.approx(w_max, 0.01) == 77.25


def test_pallet_scale_cross_check_exact_match():
    """Optical count matches pallet scale reading -> MATCH status."""
    checker = PalletScaleCrossChecker(
        nominal_carton_weight_kg=12.5,
        default_tare_weight_kg=25.0,
    )
    res = checker.cross_check(
        optical_count=6,
        gross_weight_kg=100.0,  # net 75 kg / 12.5 = 6 cartons
    )
    assert res.status == "MATCH"
    assert res.discrepancy_delta == 0
    assert res.scale_carton_count == 6
    assert res.optical_carton_count == 6
    assert res.within_tolerance is True
    assert "Corroborated" in res.diagnostic_message


def test_pallet_scale_cross_check_undercount_suspected():
    """Optical count is lower than weight indicates -> UNDERCOUNT_SUSPECTED (occluded cartons)."""
    checker = PalletScaleCrossChecker(
        nominal_carton_weight_kg=12.5,
        default_tare_weight_kg=25.0,
    )
    # Scale says 6 cartons, but camera only saw 4
    res = checker.cross_check(
        optical_count=4,
        gross_weight_kg=100.0,
    )
    assert res.status == "UNDERCOUNT_SUSPECTED"
    assert res.discrepancy_delta == -2
    assert res.scale_carton_count == 6
    assert res.optical_carton_count == 4
    assert "internal occluded cartons" in res.diagnostic_message


def test_pallet_scale_cross_check_overcount_suspected():
    """Optical count is higher than weight indicates -> OVERCOUNT_SUSPECTED (double count)."""
    checker = PalletScaleCrossChecker(
        nominal_carton_weight_kg=12.5,
        default_tare_weight_kg=25.0,
    )
    # Scale says 6 cartons, but camera saw 8
    res = checker.cross_check(
        optical_count=8,
        gross_weight_kg=100.0,
    )
    assert res.status == "OVERCOUNT_SUSPECTED"
    assert res.discrepancy_delta == 2
    assert res.scale_carton_count == 6
    assert res.optical_carton_count == 8
    assert "double-counting" in res.diagnostic_message


# ============================================================================
# 2. Carton-Level Exact Match (CLEM) per Event Evaluator Tests
# ============================================================================

def test_clem_evaluation_100_percent_pass():
    """All predicted crossing events exactly match ground truth -> CLEM 100%."""
    evaluator = CartonLevelEvaluator(match_window_sec=2.0)

    gt_events = [
        {"event_id": "gt_1", "timestamp_sec": 1.0, "carton_count": 1},
        {"event_id": "gt_2", "timestamp_sec": 5.0, "carton_count": 2},
        {"event_id": "gt_3", "timestamp_sec": 10.0, "carton_count": 1},
        {"event_id": "gt_4", "timestamp_sec": 15.0, "carton_count": 3},
    ]
    pred_events = [
        {"event_id": "p_1", "timestamp_sec": 1.1, "carton_count": 1, "decision_action": "COMMIT"},
        {"event_id": "p_2", "timestamp_sec": 5.05, "carton_count": 2, "decision_action": "COMMIT"},
        {"event_id": "p_3", "timestamp_sec": 9.95, "carton_count": 1, "decision_action": "COMMIT"},
        {"event_id": "p_4", "timestamp_sec": 15.2, "carton_count": 3, "decision_action": "COMMIT"},
    ]

    report = evaluator.evaluate(gt_events, pred_events)
    assert report.total_ground_truth_events == 4
    assert report.exact_match_events == 4
    assert report.clem_rate_pct == 100.0
    assert report.clem_target_met is True
    assert report.ambiguous_stack_events == 0
    assert report.ambiguous_stack_rate_pct == 0.0
    assert report.ambiguity_target_met is True
    assert report.passed_week5_benchmark is True
    assert report.overcount_events == 0
    assert report.undercount_events == 0
    assert report.missed_events == 0


def test_clem_evaluation_fails_below_95_percent_target():
    """Exact match rate below 95% fails Week 5 benchmark."""
    evaluator = CartonLevelEvaluator(clem_target_pct=95.0)

    # 20 ground truth events, 18 exact, 2 mismatch -> 90.0% CLEM < 95.0%
    gt_events = [{"event_id": f"gt_{i}", "timestamp_sec": i * 3.0, "carton_count": 1} for i in range(20)]
    pred_events = [
        {"event_id": f"p_{i}", "timestamp_sec": i * 3.0 + 0.1, "carton_count": 1 if i < 18 else 2, "decision_action": "COMMIT"}
        for i in range(20)
    ]

    report = evaluator.evaluate(gt_events, pred_events)
    assert report.total_ground_truth_events == 20
    assert report.exact_match_events == 18
    assert report.clem_rate_pct == 90.0
    assert report.clem_target_met is False
    assert report.passed_week5_benchmark is False
    assert report.overcount_events == 2


def test_clem_evaluation_ambiguous_stack_rate_constraint():
    """AMBIGUOUS_STACK rate >= 5% fails benchmark even if CLEM is high."""
    evaluator = CartonLevelEvaluator(clem_target_pct=95.0, max_ambiguous_rate_pct=5.0)

    # 20 events: all exact count match, but 2 flagged as AMBIGUOUS_STACK (10.0% > 5%)
    gt_events = [{"event_id": f"gt_{i}", "timestamp_sec": i * 2.0, "carton_count": 1} for i in range(20)]
    pred_events = [
        {
            "event_id": f"p_{i}",
            "timestamp_sec": i * 2.0,
            "carton_count": 1,
            "decision_action": "AMBIGUOUS_STACK" if i < 2 else "COMMIT",
        }
        for i in range(20)
    ]

    report = evaluator.evaluate(gt_events, pred_events)
    assert report.clem_rate_pct == 100.0
    assert report.clem_target_met is True
    assert report.ambiguous_stack_events == 2
    assert report.ambiguous_stack_rate_pct == 10.0
    assert report.ambiguity_target_met is False
    assert report.passed_week5_benchmark is False


def test_clem_evaluation_with_missed_and_spurious_events():
    """Correctly categorizes missed and spurious events."""
    evaluator = CartonLevelEvaluator(match_window_sec=1.5)

    gt_events = [
        {"event_id": "gt_1", "timestamp_sec": 2.0, "carton_count": 1},
        {"event_id": "gt_2", "timestamp_sec": 10.0, "carton_count": 2},  # missed
    ]
    pred_events = [
        {"event_id": "p_1", "timestamp_sec": 2.1, "carton_count": 1, "decision_action": "COMMIT"},
        {"event_id": "p_spurious", "timestamp_sec": 25.0, "carton_count": 1, "decision_action": "COMMIT"},  # spurious
    ]

    report = evaluator.evaluate(gt_events, pred_events)
    assert report.exact_match_events == 1
    assert report.missed_events == 1
    assert report.spurious_events == 1
    assert report.clem_rate_pct == 50.0
    assert len(report.event_matches) == 3


def test_clem_evaluation_against_ground_truth_session_18_csv():
    """
    Evaluates against actual ground truth dataset session_18_events.csv:
    Ground truth has 6 crossings of 1 carton each (total 6 cartons).
    """
    gt_csv_path = "data/ground_truth/session_18_events.csv"
    assert os.path.exists(gt_csv_path)

    gt_events = []
    with open(gt_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_events.append({
                "event_id": row["event_id"],
                "timestamp_sec": float(row["event_timestamp"]),
                "carton_count": int(row["carton_count"]),
                "carrying_mode": row["carrying_mode"],
                "notes": row["notes"],
            })

    assert len(gt_events) == 6

    # Simulate predictions with small realistic jitter (+/- 0.15s)
    pred_events = [
        {"event_id": f"fused_{i+1}", "timestamp_sec": ev["timestamp_sec"] + 0.1, "carton_count": 1, "decision_action": "COMMIT"}
        for i, ev in enumerate(gt_events)
    ]

    checker = PalletScaleCrossChecker(nominal_carton_weight_kg=12.5, default_tare_weight_kg=25.0)
    evaluator = CartonLevelEvaluator()
    report = evaluator.evaluate(
        ground_truth_events=gt_events,
        predicted_events=pred_events,
        pallet_scale_checker=checker,
        gross_weight_kg=100.0,
    )

    assert report.total_ground_truth_events == 6
    assert report.total_predicted_events == 6
    assert report.exact_match_events == 6
    assert report.clem_rate_pct == 100.0
    assert report.clem_target_met is True
    assert report.ambiguous_stack_events == 0
    assert report.ambiguity_target_met is True
    assert report.passed_week5_benchmark is True
    assert report.pallet_scale_audit is not None
    assert report.pallet_scale_audit["status"] == "MATCH"


# ============================================================================
# 3. Pipeline Integration Tests
# ============================================================================

def test_pipeline_pallet_scale_telemetry_integration():
    """Verifies ThreeCameraCartonPipeline pallet scale methods and reset lifecycle."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    assert pipeline.pallet_scale_audit is None

    # Set pallet scale telemetry: gross 100 kg, tare 25 kg, unit 12.5 kg -> 6 cartons
    audit = pipeline.set_pallet_scale_telemetry(gross_weight_kg=100.0, tare_weight_kg=25.0, unit_weight_kg=12.5)
    assert isinstance(audit, PalletScaleAuditResult)
    assert audit.scale_carton_count == 6
    assert pipeline.pallet_scale_audit is not None
    assert pipeline.pallet_scale_audit["scale_carton_count"] == 6

    # Test reset clears telemetry
    pipeline.reset()
    assert pipeline.pallet_scale_audit is None


# ============================================================================
# 4. API Endpoints Tests
# ============================================================================

def test_api_review_queue_empty_by_default():
    """GET /api/3angle/review_queue returns empty queue when no review events pending."""
    client = app.test_client()
    resp = client.get("/api/3angle/review_queue")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert isinstance(data["queue"], list)
    assert "total_flagged" in data
    assert "ambiguous_stack_count" in data
    assert "hold_for_review_count" in data


def test_api_review_queue_populates_flagged_events_and_audits():
    """
    Simulates flagged review events in state, confirms GET /api/3angle/review_queue,
    and performs 1-click audit approval.
    """
    app.config["TESTING"] = True
    client = app.test_client()

    # Inject mock review event into state.multi_angle_report
    mock_event = {
        "event_id": "test_flagged_001",
        "global_carton_id": "gc_flagged_001",
        "timestamp_sec": 4.5,
        "fused_cartons": 2,
        "requires_review": True,
        "decision_action": "HOLD_FOR_REVIEW",
        "decision_reason": "Conflict within 1 carton between side and flank",
        "details": {
            "event_id": "test_flagged_001",
            "timestamp_sec": 4.5,
            "carton_count": 2,
            "requires_review": True,
            "decision_action": "HOLD_FOR_REVIEW",
            "decision_reason": "Conflict within 1 carton between side and flank",
        }
    }
    state.multi_angle_report = {
        "result": {
            "events": [mock_event],
            "total_loaded_cartons": 2,
        }
    }

    # Verify queue returns the event
    resp = client.get("/api/3angle/review_queue")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total_flagged"] >= 1
    found = any(item["event_id"] == "test_flagged_001" for item in data["queue"])
    assert found is True

    # 1-click Approve
    audit_resp = client.post("/api/3angle/audit_event", json={
        "event_id": "test_flagged_001",
        "action": "approve",
    }, headers={"Authorization": "Bearer op_token_dock_1"})
    assert audit_resp.status_code == 200
    audit_data = audit_resp.get_json()
    assert audit_data["success"] is True
    assert mock_event["requires_review"] is False

    # Queue should now have cleared that event
    resp_after = client.get("/api/3angle/review_queue")
    data_after = resp_after.get_json()
    assert not any(item["event_id"] == "test_flagged_001" for item in data_after["queue"])


def test_api_pallet_scale_endpoint():
    """POST /api/3angle/pallet_scale executes weight cross-check."""
    client = app.test_client()
    state.total_count = 6

    # Test POST cross-check
    resp = client.post("/api/3angle/pallet_scale", json={
        "gross_weight_kg": 100.0,
        "tare_weight_kg": 25.0,
        "nominal_carton_weight_kg": 12.5,
        "optical_count": 6,
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["audit"]["status"] == "MATCH"
    assert data["audit"]["scale_carton_count"] == 6
    assert data["audit"]["optical_carton_count"] == 6
    assert data["audit"]["discrepancy_delta"] == 0

    # Test GET retrieves current audit
    resp_get = client.get("/api/3angle/pallet_scale")
    assert resp_get.status_code == 200
    get_data = resp_get.get_json()
    assert get_data["success"] is True
    assert get_data["audit"]["status"] == "MATCH"
