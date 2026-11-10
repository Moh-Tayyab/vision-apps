"""
Event-Level Ground Truth & Certification Evaluator (Phase 1.4 & 1.9).

Defines the Event-Level Ground Truth CSV schema and implements deterministic
temporal bipartite matching to compute True Positives (TP), False Positives (FP),
False Negatives (FN), Event-Level Recall, and Event-Level Precision.

Stage 1 Certification Gate:
  - Event-Level Recall >= 96.0%
  - Event-Level Precision >= 96.0%
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class GroundTruthEvent:
    """One physical ground-truth carton crossing event."""

    event_id: str
    timestamp_sec: float
    direction: str  # "IN" or "OUT"
    carton_count: int = 1
    carrying_mode: str = "single_worker"  # "single_worker", "dual_worker", "forklift"
    timestamp_start_sec: Optional[float] = None
    timestamp_commit_sec: Optional[float] = None
    reviewer_id: str = "unassigned"
    is_ambiguous: bool = False
    provenance: str = "unverified"  # "human_reviewed", "synthetic_fixture", "unverified"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("IN", "OUT"):
            raise ValueError(f"Invalid direction '{self.direction}'. Must be 'IN' or 'OUT'.")
        if self.carton_count < 1:
            raise ValueError(f"carton_count must be >= 1, got {self.carton_count}")
        if self.timestamp_sec < 0.0:
            raise ValueError(f"timestamp_sec must be >= 0.0, got {self.timestamp_sec}")
        if (
            self.timestamp_start_sec is not None
            and self.timestamp_commit_sec is not None
            and self.timestamp_commit_sec < self.timestamp_start_sec
        ):
            raise ValueError(
                f"timestamp_commit ({self.timestamp_commit_sec}) cannot precede timestamp_start ({self.timestamp_start_sec})"
            )


@dataclass(frozen=True)
class PredictedEvent:
    """One detected carton crossing event emitted by the counting pipeline."""

    prediction_id: str
    timestamp_sec: float
    direction: str
    carton_count: int = 1
    confidence: float = 1.0


@dataclass
class EventEvaluationMetrics:
    """Comprehensive event-level performance metrics."""

    session_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    ground_truth_count: int
    predicted_count: int
    recall: float
    precision: float
    f1_score: float
    matched_pairs: List[Tuple[str, str, float]] = field(default_factory=list)  # (gt_id, pred_id, dt)
    unmatched_gt: List[str] = field(default_factory=list)
    unmatched_pred: List[str] = field(default_factory=list)

    @property
    def passed_stage1_gate(self) -> bool:
        """Stage 1 gate requirement: Recall >= 96% and Precision >= 96%."""
        return self.recall >= 0.96 and self.precision >= 0.96

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "ground_truth_count": self.ground_truth_count,
            "predicted_count": self.predicted_count,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1_score": round(self.f1_score, 4),
            "passed_stage1_gate": self.passed_stage1_gate,
        }


class EventMatcher:
    """
    Deterministic 1-to-1 temporal bipartite event matcher.
    Associates predictions with ground truth within max_dt_sec tolerance.
    """

    def __init__(self, max_dt_sec: float = 1.0) -> None:
        self.max_dt = max_dt_sec

    def evaluate_session(
        self,
        session_id: str,
        ground_truth: Sequence[GroundTruthEvent],
        predictions: Sequence[PredictedEvent],
    ) -> EventEvaluationMetrics:
        """
        Evaluate predicted crossings against ground-truth events.
        Matches greedily by minimum time delta within tolerance window.
        """
        # Expand multi-carton events into individual item units for exact carton accounting
        gt_units: List[Tuple[str, float, str]] = []  # (unit_id, timestamp, direction)
        for gt in ground_truth:
            for c in range(gt.carton_count):
                gt_units.append((f"{gt.event_id}_{c}", gt.timestamp_sec, gt.direction))

        pred_units: List[Tuple[str, float, str]] = []
        for p in predictions:
            for c in range(p.carton_count):
                pred_units.append((f"{p.prediction_id}_{c}", p.timestamp_sec, p.direction))

        matched_gt_ids = set()
        matched_pred_ids = set()
        matched_pairs: List[Tuple[str, str, float]] = []

        # Find candidate candidate pairs: (dt, gt_id, pred_id)
        candidates = []
        for gt_id, gt_t, gt_dir in gt_units:
            for pred_id, pred_t, pred_dir in pred_units:
                if gt_dir == pred_dir:
                    dt = abs(gt_t - pred_t)
                    if dt <= self.max_dt:
                        candidates.append((dt, gt_id, pred_id))

        # Sort candidates by time delta ascending (nearest first)
        candidates.sort(key=lambda x: x[0])

        for dt, gt_id, pred_id in candidates:
            if gt_id not in matched_gt_ids and pred_id not in matched_pred_ids:
                matched_gt_ids.add(gt_id)
                matched_pred_ids.add(pred_id)
                matched_pairs.append((gt_id, pred_id, round(dt, 3)))

        tp = len(matched_pairs)
        fn = len(gt_units) - tp
        fp = len(pred_units) - tp

        unmatched_gt = [g[0] for g in gt_units if g[0] not in matched_gt_ids]
        unmatched_pred = [p[0] for p in pred_units if p[0] not in matched_pred_ids]

        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return EventEvaluationMetrics(
            session_id=session_id,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            ground_truth_count=len(gt_units),
            predicted_count=len(pred_units),
            recall=recall,
            precision=precision,
            f1_score=f1,
            matched_pairs=matched_pairs,
            unmatched_gt=unmatched_gt,
            unmatched_pred=unmatched_pred,
        )

    @staticmethod
    def validate_event_sequence(events: Sequence[GroundTruthEvent]) -> None:
        """
        Validate chronological ordering, bounds, and prevent duplicate event IDs.
        """
        seen_ids = set()
        prev_t = -1.0
        for ev in events:
            if ev.event_id in seen_ids:
                raise ValueError(f"Duplicate event_id detected: {ev.event_id}")
            seen_ids.add(ev.event_id)
            if ev.timestamp_sec < prev_t:
                raise ValueError(
                    f"Out-of-order event timestamps: {ev.event_id} (t={ev.timestamp_sec}s) follows t={prev_t}s"
                )
            prev_t = ev.timestamp_sec

    @staticmethod
    def assert_certified_provenance(events: Sequence[GroundTruthEvent]) -> None:
        """
        Enforce that synthetic test fixtures CANNOT be passed as certified ground truth.
        """
        for ev in events:
            if ev.provenance == "synthetic_fixture":
                raise ValueError(
                    f"Certification Rejection: Event {ev.event_id} has provenance 'synthetic_fixture'. "
                    "Only human-reviewed events with verified provenance are admissible for certification."
                )

    @staticmethod
    def load_ground_truth_csv(path: Path | str) -> List[GroundTruthEvent]:
        """Load ground truth events from standard CSV format."""
        events = []
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Ground truth CSV not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(
                    GroundTruthEvent(
                        event_id=str(row["event_id"]),
                        timestamp_sec=float(row["event_timestamp"]),
                        direction=str(row["direction"]).strip().upper(),
                        carton_count=int(row.get("carton_count", 1)),
                        carrying_mode=str(row.get("carrying_mode", "single_worker")),
                        timestamp_start_sec=float(row["timestamp_start"]) if row.get("timestamp_start") else None,
                        timestamp_commit_sec=float(row["timestamp_commit"]) if row.get("timestamp_commit") else None,
                        reviewer_id=str(row.get("reviewer_id", "unassigned")),
                        is_ambiguous=str(row.get("is_ambiguous", "False")).lower() in ("true", "1"),
                        provenance=str(row.get("provenance", "unverified")),
                        notes=str(row.get("notes", "")),
                    )
                )
        return events

    @staticmethod
    def save_ground_truth_csv(events: Sequence[GroundTruthEvent], path: Path | str) -> None:
        """Save ground truth events to standard CSV format."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "event_id",
                "event_timestamp",
                "direction",
                "carton_count",
                "carrying_mode",
                "timestamp_start",
                "timestamp_commit",
                "reviewer_id",
                "is_ambiguous",
                "provenance",
                "notes",
            ])
            for e in events:
                writer.writerow([
                    e.event_id,
                    e.timestamp_sec,
                    e.direction,
                    e.carton_count,
                    e.carrying_mode,
                    e.timestamp_start_sec if e.timestamp_start_sec is not None else "",
                    e.timestamp_commit_sec if e.timestamp_commit_sec is not None else "",
                    e.reviewer_id,
                    e.is_ambiguous,
                    e.provenance,
                    e.notes,
                ])


load_ground_truth_events = EventMatcher.load_ground_truth_csv

