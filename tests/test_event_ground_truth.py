"""
Unit tests for Event-Level Ground Truth Matching and Stage 1 Certification Gate (Phase 1.4 & 1.9).
"""

import pytest

from apps.truck_loading_detection.model_pipeline.event_ground_truth import (
    EventMatcher,
    GroundTruthEvent,
    PredictedEvent,
)


def test_perfect_event_matching_passes_stage1():
    """Verify that perfect predictions achieve 100% recall and precision, passing Stage 1."""
    matcher = EventMatcher(max_dt_sec=1.0)

    gt_events = [
        GroundTruthEvent("ev_01", 1.5, "IN", 1),
        GroundTruthEvent("ev_02", 3.2, "IN", 1),
        GroundTruthEvent("ev_03", 5.0, "OUT", 1),
    ]
    pred_events = [
        PredictedEvent("pred_01", 1.55, "IN", 1),
        PredictedEvent("pred_02", 3.18, "IN", 1),
        PredictedEvent("pred_03", 4.95, "OUT", 1),
    ]

    metrics = matcher.evaluate_session("sess_test", gt_events, pred_events)
    assert metrics.true_positives == 3
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.recall == 1.0
    assert metrics.precision == 1.0
    assert metrics.passed_stage1_gate is True


def test_missed_and_spurious_events():
    """Verify TP, FP, FN calculations when events are missed or spurious."""
    matcher = EventMatcher(max_dt_sec=1.0)

    gt_events = [
        GroundTruthEvent("ev_01", 1.0, "IN", 1),
        GroundTruthEvent("ev_02", 3.0, "IN", 1),
        GroundTruthEvent("ev_03", 5.0, "IN", 1),
    ]
    pred_events = [
        PredictedEvent("pred_01", 1.0, "IN", 1),   # Match ev_01 (TP)
        # Missed ev_02 (FN)
        PredictedEvent("pred_03", 5.0, "IN", 1),   # Match ev_03 (TP)
        PredictedEvent("pred_04", 7.0, "IN", 1),   # Spurious event (FP)
    ]

    metrics = matcher.evaluate_session("sess_test", gt_events, pred_events)
    assert metrics.true_positives == 2
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 1
    assert pytest.approx(metrics.recall, abs=1e-3) == 2 / 3  # 66.7%
    assert pytest.approx(metrics.precision, abs=1e-3) == 2 / 3
    assert metrics.passed_stage1_gate is False


def test_direction_mismatch_does_not_match():
    """Predicted OUT cannot match ground truth IN."""
    matcher = EventMatcher(max_dt_sec=1.0)

    gt_events = [GroundTruthEvent("ev_01", 2.0, "IN", 1)]
    pred_events = [PredictedEvent("pred_01", 2.0, "OUT", 1)]

    metrics = matcher.evaluate_session("sess_dir", gt_events, pred_events)
    assert metrics.true_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 1


def test_temporal_tolerance_boundary():
    """Events separated by > 1.0s do not match."""
    matcher = EventMatcher(max_dt_sec=1.0)

    gt_events = [GroundTruthEvent("ev_01", 2.0, "IN", 1)]
    # 1.05s apart > 1.0s
    pred_events = [PredictedEvent("pred_01", 3.05, "IN", 1)]

    metrics = matcher.evaluate_session("sess_time", gt_events, pred_events)
    assert metrics.true_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 1


def test_event_sequence_validation_and_duplicate_prevention():
    """Verify detection of duplicate event IDs, negative timestamps, and out-of-order events."""
    # Negative timestamp
    with pytest.raises(ValueError, match="timestamp_sec must be >= 0.0"):
        GroundTruthEvent("ev_neg", -1.0, "IN", 1)

    # Commit preceding start
    with pytest.raises(ValueError, match="cannot precede timestamp_start"):
        GroundTruthEvent("ev_inv", 2.0, "IN", 1, timestamp_start_sec=2.5, timestamp_commit_sec=1.5)

    # Duplicate event ID
    dup_events = [
        GroundTruthEvent("ev_01", 1.0, "IN", 1),
        GroundTruthEvent("ev_01", 2.0, "IN", 1),
    ]
    with pytest.raises(ValueError, match="Duplicate event_id detected"):
        EventMatcher.validate_event_sequence(dup_events)

    # Out of order events
    out_of_order = [
        GroundTruthEvent("ev_01", 3.0, "IN", 1),
        GroundTruthEvent("ev_02", 2.0, "IN", 1),
    ]
    with pytest.raises(ValueError, match="Out-of-order event timestamps"):
        EventMatcher.validate_event_sequence(out_of_order)


def test_synthetic_fixture_rejected_from_certified_provenance():
    """CRITICAL QA GATE: Synthetic test fixtures cannot be passed off as certified ground truth."""
    synthetic_events = [
        GroundTruthEvent("ev_01", 1.0, "IN", 1, provenance="synthetic_fixture"),
    ]
    with pytest.raises(ValueError, match="Certification Rejection:.*synthetic_fixture"):
        EventMatcher.assert_certified_provenance(synthetic_events)

    # Verified human reviewed event passes
    verified_events = [
        GroundTruthEvent("ev_01", 1.0, "IN", 1, reviewer_id="qa_lead", provenance="human_reviewed"),
    ]
    # Does not raise
    EventMatcher.assert_certified_provenance(verified_events)


def test_enhanced_ground_truth_csv_roundtrip(tmp_path):
    """Verify CSV roundtrip with reviewer ID, interval, ambiguity and provenance."""
    test_csv = tmp_path / "test_events.csv"
    orig_events = [
        GroundTruthEvent(
            event_id="test_ev_01",
            timestamp_sec=5.25,
            direction="IN",
            carton_count=2,
            carrying_mode="dual_worker",
            timestamp_start_sec=5.0,
            timestamp_commit_sec=5.5,
            reviewer_id="lead_reviewer_01",
            is_ambiguous=False,
            provenance="human_reviewed",
            notes="Clear visual crossing",
        )
    ]
    EventMatcher.save_ground_truth_csv(orig_events, test_csv)
    loaded = EventMatcher.load_ground_truth_csv(test_csv)

    assert len(loaded) == 1
    ev = loaded[0]
    assert ev.event_id == "test_ev_01"
    assert ev.timestamp_sec == 5.25
    assert ev.direction == "IN"
    assert ev.carton_count == 2
    assert ev.carrying_mode == "dual_worker"
    assert ev.timestamp_start_sec == 5.0
    assert ev.timestamp_commit_sec == 5.5
    assert ev.reviewer_id == "lead_reviewer_01"
    assert ev.is_ambiguous is False
    assert ev.provenance == "human_reviewed"

