"""
tests/test_ground_truth_reconciler.py

Unit tests for Phase 3.1 Ground Truth Reconciliation Engine.
Verifies:
  1. Strict 1-to-1 bipartite assignment (no double-matching).
  2. Bounded temporal matching tolerance (outside tolerance -> MISSED / UNMATCHED).
  3. Classification into MATCHED_AUTO_CONFIRMED, MATCHED_REVIEW_REQUIRED, MISSED.
  4. Classification of unmatched predictions into FALSE_POSITIVE, DUPLICATE_CANDIDATE, UNRESOLVED_PREDICTION.
  5. Direction consistency constraint (IN vs OUT never matches).
  6. Transparent metrics accounting and isolation of auto-confirmed vs review counts.
"""

import pytest
from apps.truck_loading_detection.three_camera_system.reconciliation_engine import (
    GroundTruthClassification,
    GroundTruthCrossing,
    GroundTruthReconciler,
    PredictionClassification,
    PredictedCrossing,
    SessionMetrics,
)


def test_reconciler_strict_one_to_one_matching():
    """Ensure that two predictions close to one GT match only one, leaving the other as duplicate candidate."""
    reconciler = GroundTruthReconciler(temporal_tolerance_sec=2.0)

    gt = [
        GroundTruthCrossing(
            event_id="gt_01",
            session_id="test_sess",
            timestamp_sec=10.0,
            direction="IN",
        )
    ]

    preds = [
        PredictedCrossing(
            event_id="pred_01",
            session_id="test_sess",
            global_carton_id="carton_01",
            timestamp_sec=10.1,
            direction="IN",
            is_auto_confirmed=True,
            participating_cameras=("cam_side", "cam_flank"),
        ),
        PredictedCrossing(
            event_id="pred_02",
            session_id="test_sess",
            global_carton_id="carton_02",
            timestamp_sec=10.3,
            direction="IN",
            is_auto_confirmed=False,
            requires_review=True,
            participating_cameras=("cam_top",),
        ),
    ]

    metrics, matched, unmatched, missed = reconciler.reconcile_session(
        ground_truth=gt,
        predictions=preds,
        session_id="test_sess",
        frames_evaluated=300,
        duration_sec=10.0,
    )

    assert len(matched) == 1
    assert matched[0].gt_event_id == "gt_01"
    assert matched[0].pred_event_id == "pred_01"
    assert matched[0].classification == GroundTruthClassification.MATCHED_AUTO_CONFIRMED

    assert len(missed) == 0
    assert len(unmatched) == 1
    assert unmatched[0].pred_event_id == "pred_02"
    assert unmatched[0].classification == PredictionClassification.DUPLICATE_CANDIDATE
    assert metrics.matched_auto_confirmed == 1
    assert metrics.duplicate_candidates == 1


def test_reconciler_temporal_tolerance_boundary():
    """Events separated by more than temporal_tolerance_sec must not match."""
    reconciler = GroundTruthReconciler(temporal_tolerance_sec=1.5)

    gt = [
        GroundTruthCrossing(
            event_id="gt_01",
            session_id="test_sess",
            timestamp_sec=5.0,
            direction="IN",
        )
    ]

    preds = [
        PredictedCrossing(
            event_id="pred_01",
            session_id="test_sess",
            global_carton_id="carton_01",
            timestamp_sec=6.6,  # delta_t = 1.6s > 1.5s tolerance
            direction="IN",
            is_auto_confirmed=True,
            participating_cameras=("cam_side", "cam_flank"),
        )
    ]

    metrics, matched, unmatched, missed = reconciler.reconcile_session(
        ground_truth=gt,
        predictions=preds,
        session_id="test_sess",
        frames_evaluated=300,
        duration_sec=10.0,
    )

    assert len(matched) == 0
    assert len(missed) == 1
    assert missed[0].event_id == "gt_01"
    assert len(unmatched) == 1
    assert unmatched[0].pred_event_id == "pred_01"
    assert unmatched[0].classification == PredictionClassification.FALSE_POSITIVE


def test_reconciler_direction_mismatch_rejected():
    """Opposing directions (IN vs OUT) must never be matched even with identical timestamps."""
    reconciler = GroundTruthReconciler(temporal_tolerance_sec=2.0)

    gt = [
        GroundTruthCrossing(
            event_id="gt_01",
            session_id="test_sess",
            timestamp_sec=12.0,
            direction="IN",
        )
    ]

    preds = [
        PredictedCrossing(
            event_id="pred_01",
            session_id="test_sess",
            global_carton_id="carton_01",
            timestamp_sec=12.0,
            direction="OUT",
            is_auto_confirmed=True,
            participating_cameras=("cam_side", "cam_flank"),
        )
    ]

    metrics, matched, unmatched, missed = reconciler.reconcile_session(
        ground_truth=gt,
        predictions=preds,
        session_id="test_sess",
    )

    assert len(matched) == 0
    assert len(missed) == 1
    assert len(unmatched) == 1


def test_reconciler_review_required_classification():
    """A matched prediction that is review-required must be classified as MATCHED_REVIEW_REQUIRED."""
    reconciler = GroundTruthReconciler(temporal_tolerance_sec=2.0)

    gt = [
        GroundTruthCrossing(
            event_id="gt_01",
            session_id="test_sess",
            timestamp_sec=8.0,
            direction="IN",
        )
    ]

    preds = [
        PredictedCrossing(
            event_id="pred_01",
            session_id="test_sess",
            global_carton_id="carton_01",
            timestamp_sec=8.1,
            direction="IN",
            is_auto_confirmed=False,
            requires_review=True,
            participating_cameras=("cam_side",),
        )
    ]

    metrics, matched, unmatched, missed = reconciler.reconcile_session(
        ground_truth=gt,
        predictions=preds,
        session_id="test_sess",
    )

    assert len(matched) == 1
    assert matched[0].classification == GroundTruthClassification.MATCHED_REVIEW_REQUIRED
    assert metrics.matched_auto_confirmed == 0
    assert metrics.matched_review_required == 1
    assert metrics.automatic_true_positives == 0
    assert metrics.automatic_recall == 0.0
    assert metrics.review_required_rate == 1.0


def test_reconciler_unresolved_prediction_classification():
    """An unconfirmed prediction far from any GT crossing must be classified as UNRESOLVED_PREDICTION."""
    reconciler = GroundTruthReconciler(temporal_tolerance_sec=2.0)

    gt = [
        GroundTruthCrossing(
            event_id="gt_01",
            session_id="test_sess",
            timestamp_sec=5.0,
            direction="IN",
        )
    ]

    preds = [
        PredictedCrossing(
            event_id="pred_01",
            session_id="test_sess",
            global_carton_id="carton_01",
            timestamp_sec=25.0,  # Far away
            direction="IN",
            is_auto_confirmed=False,
            requires_review=True,
            participating_cameras=("cam_flank",),
        )
    ]

    metrics, matched, unmatched, missed = reconciler.reconcile_session(
        ground_truth=gt,
        predictions=preds,
        session_id="test_sess",
    )

    assert len(unmatched) == 1
    assert unmatched[0].classification == PredictionClassification.UNRESOLVED_PREDICTION
    assert metrics.unresolved_predictions == 1
    assert metrics.false_positives == 0


def test_reconciler_guarantees_maximum_cardinality_over_greedy():
    """
    Proves Kuhn-Munkres achieves maximum cardinality where greedy assignment fails.
    GT1 (1.0s), GT2 (1.5s).
    Pred1 (1.1s, dt_gt1=0.1s, dt_gt2=0.4s).
    Pred2 (0.0s, dt_gt1=1.0s, dt_gt2=1.5s).
    With tolerance 1.2s:
      Greedy matches GT1 <-> Pred1 (dt=0.1s), leaving GT2 unmatched (1 match).
      Kuhn-Munkres matches GT1 <-> Pred2 (dt=1.0s) and GT2 <-> Pred1 (dt=0.4s) (2 matches).
    """
    reconciler = GroundTruthReconciler(temporal_tolerance_sec=1.2)

    gt = [
        GroundTruthCrossing(event_id="gt_01", session_id="test_sess", timestamp_sec=1.0, direction="IN"),
        GroundTruthCrossing(event_id="gt_02", session_id="test_sess", timestamp_sec=1.5, direction="IN"),
    ]

    preds = [
        PredictedCrossing(
            event_id="pred_01",
            session_id="test_sess",
            global_carton_id="c1",
            timestamp_sec=1.1,
            direction="IN",
            is_auto_confirmed=True,
        ),
        PredictedCrossing(
            event_id="pred_02",
            session_id="test_sess",
            global_carton_id="c2",
            timestamp_sec=0.0,
            direction="IN",
            is_auto_confirmed=True,
        ),
    ]

    metrics, matched, unmatched, missed = reconciler.reconcile_session(
        ground_truth=gt,
        predictions=preds,
        session_id="test_sess",
    )

    assert len(matched) == 2, f"Expected 2 matches (maximum cardinality), got {len(matched)}"
    assert len(missed) == 0
    assert len(unmatched) == 0

