"""
Phase 3.1 Ground Truth Reconciliation Engine.

Implements deterministic 1-to-1 offline bipartite event matching between ground-truth
carton crossing manifests and multi-camera predicted crossing events.

Guarantees & Principles:
1. Strict 1-to-1 matching: One ground-truth crossing matches at most one predicted event,
   and one predicted event matches at most one ground-truth crossing.
2. Physically grounded temporal tolerance: Configurable matching tolerance (default 2.0s)
   grounded in physical corridor transit speed (~1.0 m/s over 2.0m depth) and strictly bounded
   by half the minimum inter-carton crossing interval (>4.0s).
3. Non-destructive: Operates purely offline; does not mutate production databases or models.
4. Granular Event Classification:
   - Ground Truth: MATCHED_AUTO_CONFIRMED, MATCHED_REVIEW_REQUIRED, MISSED
   - Predictions: MATCHED, DUPLICATE_CANDIDATE, FALSE_POSITIVE, UNRESOLVED_PREDICTION
5. Rigorous Metric Accounting: Explains every metric, numerator, and denominator.
"""

from __future__ import annotations

import csv
import enum
import json
import logging
import math
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class GroundTruthClassification(str, enum.Enum):
    MATCHED_AUTO_CONFIRMED = "MATCHED_AUTO_CONFIRMED"
    MATCHED_REVIEW_REQUIRED = "MATCHED_REVIEW_REQUIRED"
    MISSED = "MISSED"


class PredictionClassification(str, enum.Enum):
    MATCHED = "MATCHED"
    DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    UNRESOLVED_PREDICTION = "UNRESOLVED_PREDICTION"


@dataclass(frozen=True)
class GroundTruthCrossing:
    """One physical carton crossing recorded in ground-truth manifests."""

    event_id: str
    session_id: str
    timestamp_sec: float
    direction: str  # "IN" or "OUT"
    carton_count: int = 1
    carrying_mode: str = "single_worker"
    notes: str = ""


@dataclass(frozen=True)
class PredictedCrossing:
    """One logical carton crossing event predicted by the multi-camera pipeline."""

    event_id: str
    session_id: str
    global_carton_id: str
    timestamp_sec: float
    direction: str
    carton_count: int = 1
    state: str = "COUNT_COMMITTED"
    is_auto_confirmed: bool = True
    requires_review: bool = False
    participating_cameras: Tuple[str, ...] = ()
    notes: str = ""


@dataclass
class MatchedPairRecord:
    """A verified 1-to-1 pairing between ground truth and prediction."""

    gt_event_id: str
    pred_event_id: str
    session_id: str
    classification: GroundTruthClassification
    gt_timestamp_sec: float
    pred_timestamp_sec: float
    delta_t_sec: float
    direction: str
    carton_count: int
    participating_cameras: List[str]
    is_auto_confirmed: bool
    requires_review: bool
    notes: str
    is_ambiguous: bool = False
    ambiguity_reason: str = ""


@dataclass
class UnmatchedPredictionRecord:
    """An unmatched predicted crossing with classification and diagnostic reason."""

    pred_event_id: str
    session_id: str
    global_carton_id: str
    classification: PredictionClassification
    pred_timestamp_sec: float
    direction: str
    carton_count: int
    state: str
    is_auto_confirmed: bool
    requires_review: bool
    participating_cameras: List[str]
    reason: str
    is_verified_duplicate: bool = False


@dataclass
class SessionMetrics:
    """Complete, transparent metrics for a single video session."""

    session_id: str
    frames_evaluated: int
    duration_sec: float
    ground_truth_count: int
    total_predicted_events: int
    matched_auto_confirmed: int
    matched_review_required: int
    missed_cartons: int
    false_positives: int
    duplicate_candidates: int
    unresolved_predictions: int
    verified_duplicates: int = 0
    human_approved_count: int = 0
    provenance_run_id: str = "run_default"
    evaluated_db_path: str = ""

    @property
    def automatic_true_positives(self) -> int:
        return self.matched_auto_confirmed

    @property
    def automatic_false_negatives(self) -> int:
        return self.ground_truth_count - self.matched_auto_confirmed

    @property
    def automatic_false_positives(self) -> int:
        return self.false_positives

    @property
    def automatic_precision(self) -> float:
        denom = self.automatic_true_positives + self.automatic_false_positives
        return round(float(self.automatic_true_positives / denom), 4) if denom > 0 else 0.0

    @property
    def automatic_recall(self) -> float:
        denom = self.ground_truth_count
        return round(float(self.automatic_true_positives / denom), 4) if denom > 0 else 0.0

    @property
    def review_required_rate(self) -> float:
        denom = self.total_predicted_events
        rev_count = self.matched_review_required + self.unresolved_predictions
        return round(float(rev_count / denom), 4) if denom > 0 else 0.0

    @property
    def total_count_difference(self) -> int:
        return self.matched_auto_confirmed - self.ground_truth_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "provenance_run_id": self.provenance_run_id,
            "evaluated_db_path": self.evaluated_db_path,
            "frames_evaluated": self.frames_evaluated,
            "duration_sec": round(self.duration_sec, 2),
            "ground_truth_count": self.ground_truth_count,
            "total_predicted_events": self.total_predicted_events,
            "matched_auto_confirmed": self.matched_auto_confirmed,
            "matched_review_required": self.matched_review_required,
            "missed_cartons": self.missed_cartons,
            "false_positives": self.false_positives,
            "automatic_true_positives": self.automatic_true_positives,
            "automatic_false_positives": self.automatic_false_positives,
            "automatic_false_negatives": self.automatic_false_negatives,
            "duplicate_candidates": self.duplicate_candidates,
            "verified_duplicates": self.verified_duplicates,
            "unresolved_predictions": self.unresolved_predictions,
            "automatic_precision": self.automatic_precision,
            "automatic_recall": self.automatic_recall,
            "review_required_rate": self.review_required_rate,
            "human_approved_count": self.human_approved_count,
            "total_count_difference": self.total_count_difference,
        }


class GroundTruthReconciler:
    """
    Offline Ground Truth Reconciliation Engine.
    Executes maximum-cardinality minimum-cost 1-to-1 bipartite matching between physical GT events
    and pipeline predictions via the Kuhn-Munkres (Hungarian) algorithm.
    """

    def __init__(self, temporal_tolerance_sec: float = 1.90) -> None:
        # 1.90s is strictly bounded by half the minimum inter-carton crossing interval (1.947s)
        self.tolerance_sec = float(temporal_tolerance_sec)

    @staticmethod
    def load_ground_truth(csv_path: str, session_id: str) -> List[GroundTruthCrossing]:
        """Reads ground-truth crossings from manifest CSV."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")

        crossings: List[GroundTruthCrossing] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                crossings.append(
                    GroundTruthCrossing(
                        event_id=row["event_id"].strip(),
                        session_id=session_id,
                        timestamp_sec=float(row["event_timestamp"]),
                        direction=row.get("direction", "IN").strip().upper(),
                        carton_count=int(row.get("carton_count", 1)),
                        carrying_mode=row.get("carrying_mode", "single_worker").strip(),
                        notes=row.get("notes", "").strip(),
                    )
                )
        return crossings

    @staticmethod
    def load_predictions_from_db(
        db_path: str,
        session_id: str,
    ) -> List[PredictedCrossing]:
        """Reads predicted crossings from the SQLite WAL event ledger."""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")

        predictions: List[PredictedCrossing] = []
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT event_id, session_id, global_carton_id, timestamp_sec,
                       direction, carton_count, state, is_auto_confirmed,
                       requires_review, participating_cameras_json, notes
                FROM carton_crossing_events
                WHERE session_id = ?
                ORDER BY timestamp_sec ASC;
                """,
                (session_id,),
            ).fetchall()

            for r in rows:
                cams = []
                try:
                    cams = json.loads(r["participating_cameras_json"])
                except Exception:
                    pass

                predictions.append(
                    PredictedCrossing(
                        event_id=r["event_id"],
                        session_id=r["session_id"],
                        global_carton_id=r["global_carton_id"],
                        timestamp_sec=float(r["timestamp_sec"]),
                        direction=r["direction"].upper(),
                        carton_count=int(r["carton_count"]),
                        state=r["state"],
                        is_auto_confirmed=bool(r["is_auto_confirmed"]),
                        requires_review=bool(r["requires_review"]),
                        participating_cameras=tuple(sorted(cams)),
                        notes=str(r["notes"] or ""),
                    )
                )
        finally:
            conn.close()

        return predictions

    def reconcile_session(
        self,
        ground_truth: Sequence[GroundTruthCrossing],
        predictions: Sequence[PredictedCrossing],
        session_id: str,
        frames_evaluated: int = 0,
        duration_sec: float = 0.0,
        provenance_run_id: Optional[str] = None,
        evaluated_db_path: str = "",
    ) -> Tuple[SessionMetrics, List[MatchedPairRecord], List[UnmatchedPredictionRecord], List[GroundTruthCrossing]]:
        """
        Executes mathematically optimal maximum-cardinality minimum-cost 1-to-1 bipartite event matching.
        """
        n_gt = len(ground_truth)
        n_pred = len(predictions)
        run_id = provenance_run_id or f"run_{session_id}_{uuid.uuid4().hex[:8]}"

        if n_gt == 0 or n_pred == 0:
            matched_pairs: List[MatchedPairRecord] = []
            missed_gt = list(ground_truth)
            unmatched_preds = [
                UnmatchedPredictionRecord(
                    pred_event_id=p.event_id,
                    session_id=session_id,
                    global_carton_id=p.global_carton_id,
                    classification=PredictionClassification.FALSE_POSITIVE if p.is_auto_confirmed else PredictionClassification.UNRESOLVED_PREDICTION,
                    pred_timestamp_sec=p.timestamp_sec,
                    direction=p.direction,
                    carton_count=p.carton_count,
                    state=p.state,
                    is_auto_confirmed=p.is_auto_confirmed,
                    requires_review=p.requires_review,
                    participating_cameras=list(p.participating_cameras),
                    reason="No ground-truth crossings available for matching",
                )
                for p in predictions
            ]
            metrics = SessionMetrics(
                session_id=session_id,
                frames_evaluated=frames_evaluated,
                duration_sec=duration_sec,
                ground_truth_count=n_gt,
                total_predicted_events=n_pred,
                matched_auto_confirmed=0,
                matched_review_required=0,
                missed_cartons=n_gt,
                false_positives=sum(1 for u in unmatched_preds if u.classification == PredictionClassification.FALSE_POSITIVE),
                duplicate_candidates=0,
                unresolved_predictions=sum(1 for u in unmatched_preds if u.classification == PredictionClassification.UNRESOLVED_PREDICTION),
                provenance_run_id=run_id,
                evaluated_db_path=evaluated_db_path,
            )
            return metrics, matched_pairs, unmatched_preds, missed_gt

        # Cost matrix formulation:
        # We assign reward -W for any compatible edge (direction match, dt <= tolerance)
        # and cost dt, so minimizing total cost:
        #   1. Maximizes the number of matched pairs (each match yields -W)
        #   2. Minimizes total temporal discrepancy among maximum-cardinality matches.
        # Auto-confirmed events are NOT given an artificial bias, ensuring ground-truth timing fidelity.
        W = 1000.0
        INF_COST = 1e6
        cost_matrix = np.full((n_gt, n_pred), INF_COST, dtype=float)

        for i, gt in enumerate(ground_truth):
            for j, pred in enumerate(predictions):
                if gt.direction == pred.direction:
                    dt = abs(gt.timestamp_sec - pred.timestamp_sec)
                    if dt <= self.tolerance_sec:
                        cost_matrix[i, j] = dt - W

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_gt_indices: Set[int] = set()
        matched_pred_indices: Set[int] = set()
        matched_pairs: List[MatchedPairRecord] = []

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 0.0:  # Valid assignment within tolerance
                matched_gt_indices.add(i)
                matched_pred_indices.add(j)
                gt = ground_truth[i]
                pred = predictions[j]
                dt = abs(gt.timestamp_sec - pred.timestamp_sec)

                # Ambiguity check: check if any other prediction was within 100ms of this match
                is_ambiguous = False
                ambiguity_reason = ""
                for k, other_p in enumerate(predictions):
                    if k != j and other_p.direction == gt.direction:
                        other_dt = abs(gt.timestamp_sec - other_p.timestamp_sec)
                        if other_dt <= self.tolerance_sec and abs(other_dt - dt) <= 0.10:
                            is_ambiguous = True
                            ambiguity_reason = f"Candidate '{other_p.event_id}' (dt={other_dt:.3f}s) is within 100ms of primary match (dt={dt:.3f}s)"
                            break

                classification = (
                    GroundTruthClassification.MATCHED_AUTO_CONFIRMED
                    if pred.is_auto_confirmed
                    else GroundTruthClassification.MATCHED_REVIEW_REQUIRED
                )

                matched_pairs.append(
                    MatchedPairRecord(
                        gt_event_id=gt.event_id,
                        pred_event_id=pred.event_id,
                        session_id=session_id,
                        classification=classification,
                        gt_timestamp_sec=round(gt.timestamp_sec, 3),
                        pred_timestamp_sec=round(pred.timestamp_sec, 3),
                        delta_t_sec=round(dt, 3),
                        direction=gt.direction,
                        carton_count=gt.carton_count,
                        participating_cameras=list(pred.participating_cameras),
                        is_auto_confirmed=pred.is_auto_confirmed,
                        requires_review=pred.requires_review,
                        notes=pred.notes,
                        is_ambiguous=is_ambiguous,
                        ambiguity_reason=ambiguity_reason,
                    )
                )

        # Identify missed ground-truth events
        missed_gt: List[GroundTruthCrossing] = [
            ground_truth[i] for i in range(n_gt) if i not in matched_gt_indices
        ]

        # Identify and classify unmatched predictions
        unmatched_predictions: List[UnmatchedPredictionRecord] = []
        for j in range(n_pred):
            if j not in matched_pred_indices:
                pred = predictions[j]
                # Check if this prediction occurred within corridor transit window (<= 2.0s) of any GT crossing
                near_gt = any(
                    abs(pred.timestamp_sec - gt.timestamp_sec) <= self.tolerance_sec
                    for gt in ground_truth
                )
                near_matched_pred = any(
                    abs(pred.timestamp_sec - mp.pred_timestamp_sec) <= self.tolerance_sec
                    for mp in matched_pairs
                )

                if near_gt or near_matched_pred:
                    classification = PredictionClassification.DUPLICATE_CANDIDATE
                    reason = (
                        f"Secondary unmerged candidate within {self.tolerance_sec}s corridor window of verified crossing; "
                        f"observed by {list(pred.participating_cameras)}"
                    )
                elif pred.is_auto_confirmed:
                    classification = PredictionClassification.FALSE_POSITIVE
                    reason = "Uncorrelated auto-confirmed crossing without corresponding ground-truth event"
                else:
                    classification = PredictionClassification.UNRESOLVED_PREDICTION
                    reason = (
                        f"Uncorrelated single-camera observation by {list(pred.participating_cameras)} "
                        "isolated from any physical ground truth crossing"
                    )

                unmatched_predictions.append(
                    UnmatchedPredictionRecord(
                        pred_event_id=pred.event_id,
                        session_id=session_id,
                        global_carton_id=pred.global_carton_id,
                        classification=classification,
                        pred_timestamp_sec=round(pred.timestamp_sec, 3),
                        direction=pred.direction,
                        carton_count=pred.carton_count,
                        state=pred.state,
                        is_auto_confirmed=pred.is_auto_confirmed,
                        requires_review=pred.requires_review,
                        participating_cameras=list(pred.participating_cameras),
                        reason=reason,
                    )
                )

        # Compile metrics
        matched_auto = sum(1 for m in matched_pairs if m.classification == GroundTruthClassification.MATCHED_AUTO_CONFIRMED)
        matched_rev = sum(1 for m in matched_pairs if m.classification == GroundTruthClassification.MATCHED_REVIEW_REQUIRED)
        n_missed = len(missed_gt)
        n_fp = sum(1 for u in unmatched_predictions if u.classification == PredictionClassification.FALSE_POSITIVE)
        n_dup = sum(1 for u in unmatched_predictions if u.classification == PredictionClassification.DUPLICATE_CANDIDATE)
        n_unres = sum(1 for u in unmatched_predictions if u.classification == PredictionClassification.UNRESOLVED_PREDICTION)

        metrics = SessionMetrics(
            session_id=session_id,
            frames_evaluated=frames_evaluated,
            duration_sec=duration_sec,
            ground_truth_count=n_gt,
            total_predicted_events=n_pred,
            matched_auto_confirmed=matched_auto,
            matched_review_required=matched_rev,
            missed_cartons=n_missed,
            false_positives=n_fp,
            duplicate_candidates=n_dup,
            unresolved_predictions=n_unres,
            provenance_run_id=run_id,
            evaluated_db_path=evaluated_db_path,
        )

        return metrics, matched_pairs, unmatched_predictions, missed_gt
