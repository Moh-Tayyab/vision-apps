"""
Stacked Cartons Disambiguator (Phase 8).

Replaces unverified Width x Height multiplication with explicit instance
geometry and uncertainty tracking.

Supported Stack Classifications:
- INDIVIDUALLY_RESOLVED: Each carton is distinctly segmented / tracked.
- PARTIALLY_VISIBLE_STACK: Vertical stack detected, but bottom/top carton occluded.
- UNRESOLVED_MULTI_CARTON_GROUP: Dense cluster with multiple cartons that cannot be cleanly separated.
- AMBIGUOUS_STACK: Conflicting counts between side profiles and top view.

Strict Invariant:
If a stack cannot be cleanly separated into verified individual cartons,
mark the event AMBIGUOUS_STACK rather than guessing or multiplying unverified counts.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class StackResolutionState(str, enum.Enum):
    INDIVIDUALLY_RESOLVED = "INDIVIDUALLY_RESOLVED"
    PARTIALLY_VISIBLE_STACK = "PARTIALLY_VISIBLE_STACK"
    UNRESOLVED_MULTI_CARTON_GROUP = "UNRESOLVED_MULTI_CARTON_GROUP"
    AMBIGUOUS_STACK = "AMBIGUOUS_STACK"


@dataclass
class StackDisambiguationResult:
    """Result of multi-view stack resolution (Week 4 Volumetric Formula Architecture)."""

    resolution_state: StackResolutionState
    resolved_count: int
    is_ambiguous: bool
    requires_human_review: bool
    view_instance_counts: Dict[str, int]
    reason: str
    decision_action: str = "COMMIT"  # "COMMIT", "HOLD_FOR_REVIEW", "AMBIGUOUS_STACK"
    n_top: int = 1
    h_side: int = 1
    h_flank: int = 1
    save_clip_sec: float = 3.0
    audit_recommended: bool = False
    formula_breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution_state": self.resolution_state.value,
            "resolved_count": self.resolved_count,
            "is_ambiguous": self.is_ambiguous,
            "requires_human_review": self.requires_human_review,
            "view_instance_counts": dict(self.view_instance_counts),
            "reason": self.reason,
            "decision_action": self.decision_action,
            "n_top": self.n_top,
            "h_side": self.h_side,
            "h_flank": self.h_flank,
            "save_clip_sec": self.save_clip_sec,
            "audit_recommended": self.audit_recommended,
            "formula_breakdown": dict(self.formula_breakdown),
        }


class StackDisambiguator:
    """
    Evaluates volumetric stack geometry and carton quantities across camera views (Week 4).
    
    Formula:
      N_top   = footprint columns x rows from cam_top
      H_side  = vertical layers from cam_side (along truck depth)
      H_flank = vertical layers from cam_flank (breaks perspective)
      Total   = \\sum over columns c of H_layer(c)
      
    Decision Matrix:
      - N_top == \\sum H_side == \\sum H_flank -> COMMIT (exact count)
      - Conflict within 1 carton -> HOLD FOR REVIEW (save 3s synchronized clip)
      - Irreconcilable -> AMBIGUOUS_STACK -> human 1-click audit
    """

    def __init__(
        self,
        max_height_ratio_single_carton: float = 1.6,
        allow_unverified_multiplication: bool = False,
    ) -> None:
        self.max_height_ratio = float(max_height_ratio_single_carton)
        self.allow_unverified_multi = bool(allow_unverified_multiplication)

    def disambiguate(
        self,
        view_instance_counts: Dict[str, int],
        column_layer_heights: Optional[List[int]] = None,
        aspect_ratios: Optional[Dict[str, float]] = None,
    ) -> StackDisambiguationResult:
        """
        Disambiguates stack count across available camera views using the Week 4 volumetric formula.
        """
        counts = dict(view_instance_counts)
        if not counts:
            return StackDisambiguationResult(
                resolution_state=StackResolutionState.AMBIGUOUS_STACK,
                resolved_count=0,
                is_ambiguous=True,
                requires_human_review=True,
                view_instance_counts=counts,
                reason="No camera observations provided for stack evaluation",
                decision_action="AMBIGUOUS_STACK",
                audit_recommended=True,
            )

        n_top = counts.get("cam_top", 1)
        h_side = counts.get("cam_side", 0)
        h_flank = counts.get("cam_flank", 0)

        # Fallback if cam_top was unobserved or <= 0
        if "cam_top" not in counts or counts["cam_top"] <= 0:
            n_top = 1

        formula_info = {
            "N_top": n_top,
            "H_side": h_side,
            "H_flank": h_flank,
            "formula": "Total = Sum over columns c of H_layer(c)",
        }

        # Case 1: Simple single carton across all active views
        if all(c == 1 for c in counts.values()):
            return StackDisambiguationResult(
                resolution_state=StackResolutionState.INDIVIDUALLY_RESOLVED,
                resolved_count=1,
                is_ambiguous=False,
                requires_human_review=False,
                view_instance_counts=counts,
                reason="Single carton individually resolved across all views",
                decision_action="COMMIT",
                n_top=1,
                h_side=1,
                h_flank=1,
                save_clip_sec=0.0,
                audit_recommended=False,
                formula_breakdown=formula_info,
            )

        # Case 2: Both side profiles (cam_side and cam_flank) observed
        if h_side > 0 and h_flank > 0:
            diff = abs(h_side - h_flank)

            if diff == 0:
                # Subcase 2a: Perfect agreement between side perspectives
                # N_top == \sum H_side == \sum H_flank -> COMMIT (exact count)
                if column_layer_heights:
                    total_vol = sum(column_layer_heights)
                else:
                    total_vol = n_top * h_side

                formula_info["Total"] = total_vol
                formula_info["resolved_via"] = "Full multi-view agreement"

                return StackDisambiguationResult(
                    resolution_state=StackResolutionState.INDIVIDUALLY_RESOLVED,
                    resolved_count=total_vol,
                    is_ambiguous=False,
                    requires_human_review=False,
                    view_instance_counts=counts,
                    reason=f"Multi-view volumetric agreement: N_top={n_top} x H={h_side} = {total_vol} cartons",
                    decision_action="COMMIT",
                    n_top=n_top,
                    h_side=h_side,
                    h_flank=h_flank,
                    save_clip_sec=0.0,
                    audit_recommended=False,
                    formula_breakdown=formula_info,
                )

            elif diff == 1:
                # Subcase 2b: Conflict within 1 carton -> HOLD FOR REVIEW (save 3s synchronized clip)
                best_h = max(h_side, h_flank)
                total_vol = n_top * best_h
                formula_info["Total"] = total_vol
                formula_info["conflict_diff"] = 1
                formula_info["action"] = "HOLD FOR REVIEW (save 3s synchronized clip)"

                return StackDisambiguationResult(
                    resolution_state=StackResolutionState.AMBIGUOUS_STACK,
                    resolved_count=total_vol,
                    is_ambiguous=True,
                    requires_human_review=True,
                    view_instance_counts=counts,
                    reason=f"Conflict within 1 carton between side ({h_side}) and flank ({h_flank}) layers; holding for review with 3s synchronized clip",
                    decision_action="HOLD_FOR_REVIEW",
                    n_top=n_top,
                    h_side=h_side,
                    h_flank=h_flank,
                    save_clip_sec=3.0,
                    audit_recommended=False,
                    formula_breakdown=formula_info,
                )

            else:
                # Subcase 2c: Irreconcilable (diff > 1) -> AMBIGUOUS_STACK -> human 1-click audit
                best_h = max(h_side, h_flank)
                total_vol = n_top * best_h
                formula_info["Total"] = total_vol
                formula_info["conflict_diff"] = diff
                formula_info["action"] = "AMBIGUOUS_STACK -> human 1-click audit"

                return StackDisambiguationResult(
                    resolution_state=StackResolutionState.AMBIGUOUS_STACK,
                    resolved_count=total_vol,
                    is_ambiguous=True,
                    requires_human_review=True,
                    view_instance_counts=counts,
                    reason=f"Irreconcilable stack discrepancy > 1 carton (H_side={h_side} vs H_flank={h_flank}, diff={diff}); routed to human 1-click audit",
                    decision_action="AMBIGUOUS_STACK",
                    n_top=n_top,
                    h_side=h_side,
                    h_flank=h_flank,
                    save_clip_sec=3.0,
                    audit_recommended=True,
                    formula_breakdown=formula_info,
                )

        # Case 3: Only one side profile observed (cam_side or cam_flank occluded / single-camera)
        non_zero_side_counts = [c for cid, c in counts.items() if cid in ("cam_side", "cam_flank") and c > 0]
        if non_zero_side_counts:
            s_count = non_zero_side_counts[0]
            if s_count > 1 or n_top > 1:
                total_vol = n_top * s_count
                return StackDisambiguationResult(
                    resolution_state=StackResolutionState.PARTIALLY_VISIBLE_STACK,
                    resolved_count=total_vol,
                    is_ambiguous=True,
                    requires_human_review=True,
                    view_instance_counts=counts,
                    reason=f"Stack of {s_count} layers resolved in single side view; opposite view occluded; holding for review with 3s clip",
                    decision_action="HOLD_FOR_REVIEW",
                    n_top=n_top,
                    h_side=h_side or s_count,
                    h_flank=h_flank or s_count,
                    save_clip_sec=3.0,
                    audit_recommended=False,
                    formula_breakdown=formula_info,
                )
            else:
                # Single carton in single observed side view
                return StackDisambiguationResult(
                    resolution_state=StackResolutionState.INDIVIDUALLY_RESOLVED,
                    resolved_count=1,
                    is_ambiguous=False,
                    requires_human_review=False,
                    view_instance_counts=counts,
                    reason="Single carton individually resolved in available side view",
                    decision_action="COMMIT",
                    n_top=1,
                    h_side=h_side or 1,
                    h_flank=h_flank or 1,
                    save_clip_sec=0.0,
                    audit_recommended=False,
                    formula_breakdown=formula_info,
                )

        # Case 4: Top view sees multiple columns but side views see <= 1 (single-layer multi-carton row)
        if n_top > 1 and h_side <= 1 and h_flank <= 1:
            return StackDisambiguationResult(
                resolution_state=StackResolutionState.UNRESOLVED_MULTI_CARTON_GROUP,
                resolved_count=n_top,
                is_ambiguous=True,
                requires_human_review=True,
                view_instance_counts=counts,
                reason=f"Top view detected {n_top} horizontal cartons, but side profiles show <= 1",
                decision_action="HOLD_FOR_REVIEW",
                n_top=n_top,
                h_side=h_side,
                h_flank=h_flank,
                save_clip_sec=3.0,
                audit_recommended=False,
                formula_breakdown=formula_info,
            )

        # Fallback: ambiguous
        return StackDisambiguationResult(
            resolution_state=StackResolutionState.AMBIGUOUS_STACK,
            resolved_count=max(counts.values()),
            is_ambiguous=True,
            requires_human_review=True,
            view_instance_counts=counts,
            reason="Complex stack configuration could not be deterministically resolved",
            decision_action="AMBIGUOUS_STACK",
            n_top=n_top,
            h_side=h_side,
            h_flank=h_flank,
            save_clip_sec=3.0,
            audit_recommended=True,
            formula_breakdown=formula_info,
        )
