"""
Week 5: Evaluation & Deployment Module for Three-Camera Carton Counting.

Core Responsibilities:
1. Metric Formulation:
   - Carton-Level Exact Match per Event (CLEM), NOT bounding box mAP.
   - Target: >=95% exact match on visible cartons per event (y_hat == y_true).
   - Target: AMBIGUOUS_STACK rate < 5%.
2. Pallet Scale Ground-Truth Cross-Check:
   - N_pallet = round((W_loaded - W_tare) / w_carton)
   - Discrepancy diagnosis: Delta = Y_optical - N_pallet
3. Review Queue Management & Auditing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("three_camera_system.evaluator")


# ============================================================================
# 1. Pallet Scale Cross-Check Model
# ============================================================================

@dataclass
class PalletScaleReading:
    """Represents a reading from an industrial dock floor pallet scale."""
    gross_weight_kg: float
    tare_weight_kg: float
    nominal_carton_weight_kg: float
    weight_tolerance_pct: float = 3.0  # 3% tolerance for carton fill / packaging variance

    @property
    def net_weight_kg(self) -> float:
        """Net cargo weight: W_loaded - W_tare."""
        return max(0.0, self.gross_weight_kg - self.tare_weight_kg)

    @property
    def estimated_cartons(self) -> int:
        """
        N_pallet = round((W_loaded - W_tare) / w_carton)
        """
        if self.nominal_carton_weight_kg <= 0:
            return 0
        return int(round(self.net_weight_kg / self.nominal_carton_weight_kg))

    @property
    def expected_weight_range(self) -> Tuple[float, float]:
        """Expected weight range based on estimated cartons and tolerance."""
        nom = self.estimated_cartons * self.nominal_carton_weight_kg
        delta = nom * (self.weight_tolerance_pct / 100.0)
        return (max(0.0, nom - delta), nom + delta)


@dataclass
class PalletScaleAuditResult:
    """Audit comparison between optical multi-angle count and pallet scale ground truth."""
    optical_carton_count: int
    scale_carton_count: int
    net_weight_kg: float
    nominal_carton_weight_kg: float
    discrepancy_delta: int  # optical - scale
    status: str  # "MATCH", "UNDERCOUNT_SUSPECTED", "OVERCOUNT_SUSPECTED"
    weight_error_pct: float
    within_tolerance: bool
    diagnostic_message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PalletScaleCrossChecker:
    """
    Ground-truth cross-check model using pallet scale weight telemetry:
        N_pallet = round((W_loaded - W_tare) / w_carton)
    """

    def __init__(
        self,
        nominal_carton_weight_kg: float = 12.5,
        default_tare_weight_kg: float = 25.0,  # Standard wooden pallet weight
        weight_tolerance_pct: float = 3.0,
    ) -> None:
        self.nominal_carton_weight_kg = float(nominal_carton_weight_kg)
        self.default_tare_weight_kg = float(default_tare_weight_kg)
        self.weight_tolerance_pct = float(weight_tolerance_pct)

    def compute_scale_cartons(
        self,
        gross_weight_kg: float,
        tare_weight_kg: Optional[float] = None,
        unit_weight_kg: Optional[float] = None,
    ) -> int:
        """
        Calculates N_pallet = round((W_loaded - W_tare) / w_carton).
        """
        tare = self.default_tare_weight_kg if tare_weight_kg is None else float(tare_weight_kg)
        unit = self.nominal_carton_weight_kg if unit_weight_kg is None else float(unit_weight_kg)
        reading = PalletScaleReading(
            gross_weight_kg=float(gross_weight_kg),
            tare_weight_kg=tare,
            nominal_carton_weight_kg=unit,
            weight_tolerance_pct=self.weight_tolerance_pct,
        )
        return reading.estimated_cartons

    def cross_check(
        self,
        optical_count: int,
        gross_weight_kg: float,
        tare_weight_kg: Optional[float] = None,
        unit_weight_kg: Optional[float] = None,
    ) -> PalletScaleAuditResult:
        """
        Compares total optical count against pallet scale reading.
        """
        tare = self.default_tare_weight_kg if tare_weight_kg is None else float(tare_weight_kg)
        unit = self.nominal_carton_weight_kg if unit_weight_kg is None else float(unit_weight_kg)

        reading = PalletScaleReading(
            gross_weight_kg=float(gross_weight_kg),
            tare_weight_kg=tare,
            nominal_carton_weight_kg=unit,
            weight_tolerance_pct=self.weight_tolerance_pct,
        )
        scale_n = reading.estimated_cartons
        net_w = reading.net_weight_kg
        delta = optical_count - scale_n

        expected_w = optical_count * unit
        w_err_pct = (
            abs(net_w - expected_w) / expected_w * 100.0 if expected_w > 0 else 0.0
        )
        within_tol = (w_err_pct <= self.weight_tolerance_pct and delta == 0) if expected_w > 0 else (net_w == 0)

        if delta == 0:
            status = "MATCH"
            diag = (
                f"Corroborated: Optical count ({optical_count}) exactly matches "
                f"pallet scale estimate ({scale_n} cartons at net {net_w:.1f}kg)."
            )
        elif delta < 0:
            status = "UNDERCOUNT_SUSPECTED"
            diag = (
                f"Scale Invariant Alert: Optical count ({optical_count}) is {abs(delta)} fewer "
                f"than pallet scale estimate ({scale_n} cartons). "
                f"Check for internal occluded cartons hidden inside multi-layer stack."
            )
        else:
            status = "OVERCOUNT_SUSPECTED"
            diag = (
                f"Scale Invariant Alert: Optical count ({optical_count}) exceeds "
                f"pallet scale estimate ({scale_n} cartons) by {delta}. "
                f"Check for double-counting, ghost tracklets, or worker step-back."
            )

        return PalletScaleAuditResult(
            optical_carton_count=optical_count,
            scale_carton_count=scale_n,
            net_weight_kg=round(net_w, 2),
            nominal_carton_weight_kg=round(unit, 2),
            discrepancy_delta=delta,
            status=status,
            weight_error_pct=round(w_err_pct, 2),
            within_tolerance=within_tol,
            diagnostic_message=diag,
        )


# ============================================================================
# 2. Carton-Level Exact Match (CLEM) per Event Evaluator
# ============================================================================

@dataclass
class EventMatchResult:
    """Represents evaluation comparison for a single transit crossing event."""
    ground_truth_event_id: str
    ground_truth_timestamp_sec: float
    ground_truth_count: int
    predicted_event_id: Optional[str]
    predicted_timestamp_sec: Optional[float]
    predicted_count: Optional[int]
    is_matched: bool
    is_exact_match: bool
    count_delta: int  # predicted - ground_truth
    error_type: str  # "EXACT", "OVERCOUNT", "UNDERCOUNT", "MISSED", "SPURIOUS"
    decision_action: Optional[str] = None  # "COMMIT", "HOLD_FOR_REVIEW", "AMBIGUOUS_STACK"
    carrying_mode: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Week5EvaluationReport:
    """
    Week 5 Official Evaluation Report:
    - Metric: carton-level exact match per event (not mAP).
    - Target: >=95% exact match on visible cartons; AMBIGUOUS_STACK rate <5%.
    """
    total_ground_truth_events: int
    total_ground_truth_cartons: int
    total_predicted_events: int
    total_predicted_cartons: int

    # CLEM Metrics
    exact_match_events: int
    clem_rate_pct: float  # >= 95.0%

    # Ambiguity Metrics
    ambiguous_stack_events: int
    ambiguous_stack_rate_pct: float  # < 5.0%

    # Hold For Review Metrics
    hold_for_review_events: int

    # Target & Threshold Metrics with Defaults
    target_clem_pct: float = 95.0
    clem_target_met: bool = False
    target_ambiguous_rate_pct: float = 5.0
    ambiguity_target_met: bool = False
    hold_for_review_rate_pct: float = 0.0

    # Error Breakdown
    overcount_events: int = 0
    undercount_events: int = 0
    missed_events: int = 0
    spurious_events: int = 0
    net_carton_discrepancy: int = 0

    # Overall Week 5 Pass Status
    passed_week5_benchmark: bool = False

    # Event-level breakdown
    event_matches: List[EventMatchResult] = field(default_factory=list)

    # Optional pallet scale cross-check
    pallet_scale_audit: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CartonLevelEvaluator:
    """
    Evaluator computing Carton-Level Exact Match (CLEM) per event (not mAP).
    Target:
      - CLEM >= 95% on visible cartons
      - AMBIGUOUS_STACK rate < 5%
    """

    def __init__(
        self,
        match_window_sec: float = 2.0,
        clem_target_pct: float = 95.0,
        max_ambiguous_rate_pct: float = 5.0,
    ) -> None:
        self.match_window_sec = float(match_window_sec)
        self.clem_target_pct = float(clem_target_pct)
        self.max_ambiguous_rate_pct = float(max_ambiguous_rate_pct)

    def evaluate(
        self,
        ground_truth_events: List[Dict[str, Any]],
        predicted_events: List[Dict[str, Any]],
        pallet_scale_checker: Optional[PalletScaleCrossChecker] = None,
        gross_weight_kg: Optional[float] = None,
        tare_weight_kg: Optional[float] = None,
    ) -> Week5EvaluationReport:
        """
        Executes event-level exact match evaluation.
        """
        # Sort both lists chronologically
        gt_sorted = sorted(
            ground_truth_events,
            key=lambda e: float(e.get("event_timestamp") or e.get("timestamp_sec") or 0.0),
        )
        pred_sorted = sorted(
            predicted_events,
            key=lambda e: float(e.get("timestamp_sec") or e.get("event_timestamp") or 0.0),
        )

        matched_preds = set()
        event_matches: List[EventMatchResult] = []

        total_gt_cartons = sum(
            int(e.get("carton_count") or e.get("fused_cartons") or 1) for e in gt_sorted
        )
        total_pred_cartons = sum(
            int(e.get("carton_count") or e.get("fused_cartons") or 1) for e in pred_sorted
        )

        exact_matches = 0
        overcounts = 0
        undercounts = 0
        missed = 0

        # Match GT events to predicted events within temporal window
        for gt in gt_sorted:
            gt_id = str(gt.get("event_id", f"gt_ev_{len(event_matches)+1}"))
            gt_t = float(gt.get("event_timestamp") or gt.get("timestamp_sec") or 0.0)
            gt_cnt = int(gt.get("carton_count") or gt.get("fused_cartons") or 1)
            carrying_mode = gt.get("carrying_mode")
            notes = gt.get("notes")

            # Find best temporal candidate in pred_sorted not yet matched
            best_idx = None
            best_dt = float("inf")
            for idx, pred in enumerate(pred_sorted):
                if idx in matched_preds:
                    continue
                pred_t = float(pred.get("timestamp_sec") or pred.get("event_timestamp") or 0.0)
                dt = abs(gt_t - pred_t)
                if dt <= self.match_window_sec and dt < best_dt:
                    best_dt = dt
                    best_idx = idx

            if best_idx is not None:
                matched_preds.add(best_idx)
                pred = pred_sorted[best_idx]
                p_id = str(pred.get("event_id", f"pred_ev_{best_idx+1}"))
                p_t = float(pred.get("timestamp_sec") or pred.get("event_timestamp") or 0.0)
                p_cnt = int(pred.get("carton_count") or pred.get("fused_cartons") or 1)
                p_action = pred.get("decision_action") or pred.get("details", {}).get("decision_action")

                delta = p_cnt - gt_cnt
                is_exact = (delta == 0)

                if is_exact:
                    exact_matches += 1
                    err_type = "EXACT"
                elif delta > 0:
                    overcounts += 1
                    err_type = "OVERCOUNT"
                else:
                    undercounts += 1
                    err_type = "UNDERCOUNT"

                event_matches.append(
                    EventMatchResult(
                        ground_truth_event_id=gt_id,
                        ground_truth_timestamp_sec=gt_t,
                        ground_truth_count=gt_cnt,
                        predicted_event_id=p_id,
                        predicted_timestamp_sec=p_t,
                        predicted_count=p_cnt,
                        is_matched=True,
                        is_exact_match=is_exact,
                        count_delta=delta,
                        error_type=err_type,
                        decision_action=p_action,
                        carrying_mode=carrying_mode,
                        notes=notes,
                    )
                )
            else:
                # Missed GT event
                missed += 1
                event_matches.append(
                    EventMatchResult(
                        ground_truth_event_id=gt_id,
                        ground_truth_timestamp_sec=gt_t,
                        ground_truth_count=gt_cnt,
                        predicted_event_id=None,
                        predicted_timestamp_sec=None,
                        predicted_count=None,
                        is_matched=False,
                        is_exact_match=False,
                        count_delta=-gt_cnt,
                        error_type="MISSED",
                        decision_action=None,
                        carrying_mode=carrying_mode,
                        notes=notes,
                    )
                )

        # Spurious predicted events
        spurious = 0
        for idx, pred in enumerate(pred_sorted):
            if idx not in matched_preds:
                spurious += 1
                p_id = str(pred.get("event_id", f"pred_ev_{idx+1}"))
                p_t = float(pred.get("timestamp_sec") or pred.get("event_timestamp") or 0.0)
                p_cnt = int(pred.get("carton_count") or pred.get("fused_cartons") or 1)
                p_action = pred.get("decision_action") or pred.get("details", {}).get("decision_action")

                event_matches.append(
                    EventMatchResult(
                        ground_truth_event_id=f"unmatched_spurious_{spurious}",
                        ground_truth_timestamp_sec=-1.0,
                        ground_truth_count=0,
                        predicted_event_id=p_id,
                        predicted_timestamp_sec=p_t,
                        predicted_count=p_cnt,
                        is_matched=False,
                        is_exact_match=False,
                        count_delta=p_cnt,
                        error_type="SPURIOUS",
                        decision_action=p_action,
                    )
                )

        n_gt_events = len(gt_sorted)
        clem_rate = (exact_matches / n_gt_events * 100.0) if n_gt_events > 0 else 0.0
        clem_target_met = clem_rate >= self.clem_target_pct

        # Tally stack ambiguity decisions across predicted events
        ambiguous_cnt = 0
        hold_review_cnt = 0
        for p in pred_sorted:
            act = p.get("decision_action") or p.get("details", {}).get("decision_action")
            res_state = p.get("resolution_state") or p.get("details", {}).get("resolution_state")
            if act == "AMBIGUOUS_STACK" or res_state == "AMBIGUOUS_STACK":
                ambiguous_cnt += 1
            elif act == "HOLD_FOR_REVIEW" or res_state == "HOLD_FOR_REVIEW":
                hold_review_cnt += 1

        n_pred_events = len(pred_sorted)
        ambiguous_rate = (ambiguous_cnt / n_pred_events * 100.0) if n_pred_events > 0 else 0.0
        hold_review_rate = (hold_review_cnt / n_pred_events * 100.0) if n_pred_events > 0 else 0.0
        ambiguity_target_met = ambiguous_rate < self.max_ambiguous_rate_pct

        passed_benchmark = clem_target_met and ambiguity_target_met

        # Pallet scale cross-check if provided
        pallet_audit = None
        if pallet_scale_checker and gross_weight_kg is not None:
            res = pallet_scale_checker.cross_check(
                optical_count=total_pred_cartons,
                gross_weight_kg=gross_weight_kg,
                tare_weight_kg=tare_weight_kg,
            )
            pallet_audit = res.to_dict()

        return Week5EvaluationReport(
            total_ground_truth_events=n_gt_events,
            total_ground_truth_cartons=total_gt_cartons,
            total_predicted_events=n_pred_events,
            total_predicted_cartons=total_pred_cartons,
            exact_match_events=exact_matches,
            clem_rate_pct=round(clem_rate, 2),
            target_clem_pct=self.clem_target_pct,
            clem_target_met=clem_target_met,
            ambiguous_stack_events=ambiguous_cnt,
            ambiguous_stack_rate_pct=round(ambiguous_rate, 2),
            target_ambiguous_rate_pct=self.max_ambiguous_rate_pct,
            ambiguity_target_met=ambiguity_target_met,
            hold_for_review_events=hold_review_cnt,
            hold_for_review_rate_pct=round(hold_review_rate, 2),
            overcount_events=overcounts,
            undercount_events=undercounts,
            missed_events=missed,
            spurious_events=spurious,
            net_carton_discrepancy=total_pred_cartons - total_gt_cartons,
            passed_week5_benchmark=passed_benchmark,
            event_matches=event_matches,
            pallet_scale_audit=pallet_audit,
        )
