"""
Consensus Evidence Policy (Phase 7).

Evaluates cross-camera evidence and assigns a consensus status and decision.

Supported Consensus Levels:
- THREE_CAMERA_UNANIMOUS: All 3 cameras detected and agreed.
- TWO_CAMERA_AGREEMENT: 2 cameras observed and agreed; 1 camera missed or occluded.
- SINGLE_CAMERA_OBSERVATION: Only 1 camera had line of sight.
- CONFLICTING_OR_UNCERTAIN: Discrepancy in trajectory, direction, or ambiguity.

Strict Principle:
Does not treat uncalibrated heuristic scores as true statistical probabilities.
Records complete audit trail (supporting cameras, missing cameras, quality).
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)


class ConsensusLevel(str, enum.Enum):
    THREE_CAMERA_UNANIMOUS = "THREE_CAMERA_UNANIMOUS"
    TWO_CAMERA_AGREEMENT = "TWO_CAMERA_AGREEMENT"
    SINGLE_CAMERA_OBSERVATION = "SINGLE_CAMERA_OBSERVATION"
    CONFLICTING_OR_UNCERTAIN = "CONFLICTING_OR_UNCERTAIN"


class DecisionAction(str, enum.Enum):
    COMMIT = "COMMIT"
    HOLD_FOR_REVIEW = "HOLD_FOR_REVIEW"
    SUPPRESS_AS_NOISE = "SUPPRESS_AS_NOISE"


@dataclass
class ConsensusVerdict:
    """Consensus decision and complete supporting evidence record."""

    level: ConsensusLevel
    action: DecisionAction
    supporting_cameras: List[str]
    missing_cameras: List[str]
    mean_confidence: float
    evidence_score: float  # Uncalibrated composite heuristic metric [0.0 - 1.0]
    decision_reason: str
    is_statistically_calibrated: bool = False  # Explicitly False: heuristic score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "action": self.action.value,
            "supporting_cameras": list(self.supporting_cameras),
            "missing_cameras": list(self.missing_cameras),
            "mean_confidence": round(self.mean_confidence, 4),
            "evidence_score": round(self.evidence_score, 4),
            "decision_reason": self.decision_reason,
            "is_statistically_calibrated": self.is_statistically_calibrated,
        }


class ThreeCameraConsensusPolicy:
    """
    Evaluates evidence from 3 camera perspectives (top, side, flank).
    """

    def __init__(
        self,
        min_single_camera_conf: float = 0.50,
        min_two_camera_conf: float = 0.25,
        all_camera_ids: Optional[Sequence[str]] = None,
        conf_threshold: Optional[float] = None,
    ) -> None:
        if conf_threshold is not None:
            self.min_single_conf = float(conf_threshold)
        else:
            self.min_single_conf = float(min_single_camera_conf)
        self.min_two_camera_conf = float(min_two_camera_conf)
        self.all_cameras = set(all_camera_ids) if all_camera_ids else {"cam_top", "cam_side", "cam_flank"}

    def evaluate(
        self,
        observed_cameras: Sequence[str],
        confidences: Sequence[float],
        is_ambiguous: bool = False,
        has_direction_conflict: bool = False,
    ) -> ConsensusVerdict:
        """
        Computes consensus verdict for a candidate carton event.
        """
        seen_set = set(observed_cameras)
        missing = sorted(list(self.all_cameras - seen_set))
        supporting = sorted(list(seen_set))
        n_supporting = len(supporting)

        mean_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

        if has_direction_conflict:
            return ConsensusVerdict(
                level=ConsensusLevel.CONFLICTING_OR_UNCERTAIN,
                action=DecisionAction.HOLD_FOR_REVIEW,
                supporting_cameras=supporting,
                missing_cameras=missing,
                mean_confidence=mean_conf,
                evidence_score=0.20,
                decision_reason="Cameras disagree on crossing trajectory direction",
            )

        if is_ambiguous:
            return ConsensusVerdict(
                level=ConsensusLevel.CONFLICTING_OR_UNCERTAIN,
                action=DecisionAction.HOLD_FOR_REVIEW,
                supporting_cameras=supporting,
                missing_cameras=missing,
                mean_confidence=mean_conf,
                evidence_score=0.35,
                decision_reason="Cluster association flagged as spatially or temporally ambiguous",
            )

        if n_supporting >= 3:
            # 3 cameras observed
            score = min(0.99, mean_conf * 1.05)
            return ConsensusVerdict(
                level=ConsensusLevel.THREE_CAMERA_UNANIMOUS,
                action=DecisionAction.COMMIT,
                supporting_cameras=supporting,
                missing_cameras=missing,
                mean_confidence=mean_conf,
                evidence_score=score,
                decision_reason="All 3 camera perspectives confirm physical carton crossing",
            )

        elif n_supporting == 2:
            # 2 cameras observed, 1 missing (e.g. occlusion)
            if mean_conf >= self.min_two_camera_conf:
                score = min(0.92, mean_conf * 0.95)
                return ConsensusVerdict(
                    level=ConsensusLevel.TWO_CAMERA_AGREEMENT,
                    action=DecisionAction.COMMIT,
                    supporting_cameras=supporting,
                    missing_cameras=missing,
                    mean_confidence=mean_conf,
                    evidence_score=score,
                    decision_reason=f"Two cameras confirm crossing; '{missing[0]}' missed observation (probable occlusion)",
                )
            elif mean_conf >= 0.15:
                # Low confidence two-camera observation: hold for human review
                score = min(0.60, mean_conf * 0.75)
                return ConsensusVerdict(
                    level=ConsensusLevel.CONFLICTING_OR_UNCERTAIN,
                    action=DecisionAction.HOLD_FOR_REVIEW,
                    supporting_cameras=supporting,
                    missing_cameras=missing,
                    mean_confidence=mean_conf,
                    evidence_score=score,
                    decision_reason=f"Two cameras observed crossing but mean confidence {mean_conf:.3f} is below auto-confirm threshold {self.min_two_camera_conf:.2f}; routed to review",
                )
            else:
                # Sub-0.15 noise: suppress
                return ConsensusVerdict(
                    level=ConsensusLevel.CONFLICTING_OR_UNCERTAIN,
                    action=DecisionAction.SUPPRESS_AS_NOISE,
                    supporting_cameras=supporting,
                    missing_cameras=missing,
                    mean_confidence=mean_conf,
                    evidence_score=mean_conf * 0.30,
                    decision_reason=f"Two cameras detected noise below minimum confidence threshold 0.15 (mean: {mean_conf:.3f})",
                )

        elif n_supporting == 1:
            # Single camera observation
            if mean_conf >= self.min_single_conf:
                # High confidence single view allowed with review flag
                return ConsensusVerdict(
                    level=ConsensusLevel.SINGLE_CAMERA_OBSERVATION,
                    action=DecisionAction.HOLD_FOR_REVIEW,
                    supporting_cameras=supporting,
                    missing_cameras=missing,
                    mean_confidence=mean_conf,
                    evidence_score=mean_conf * 0.65,
                    decision_reason=f"Only camera '{supporting[0]}' detected carton with high confidence; views from {missing} absent",
                )
            else:
                # Low confidence single view suppressed
                return ConsensusVerdict(
                    level=ConsensusLevel.SINGLE_CAMERA_OBSERVATION,
                    action=DecisionAction.SUPPRESS_AS_NOISE,
                    supporting_cameras=supporting,
                    missing_cameras=missing,
                    mean_confidence=mean_conf,
                    evidence_score=mean_conf * 0.30,
                    decision_reason=f"Single camera '{supporting[0]}' confidence {mean_conf:.2f} below threshold {self.min_single_conf:.2f}",
                )

        else:
            return ConsensusVerdict(
                level=ConsensusLevel.CONFLICTING_OR_UNCERTAIN,
                action=DecisionAction.SUPPRESS_AS_NOISE,
                supporting_cameras=[],
                missing_cameras=missing,
                mean_confidence=0.0,
                evidence_score=0.0,
                decision_reason="No camera observations provided",
            )
