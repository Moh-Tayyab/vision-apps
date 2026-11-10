"""
Canonical Class Contract for Phase 1.

Locks exactly two detector classes:
  0: carton  (Object of interest for counting and inventory)
  1: person  (Strictly for occluder-mask generation; NEVER a carton event)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

CLASS_CARTON: int = 0
CLASS_PERSON: int = 1

CLASS_NAMES: Dict[int, str] = {
    CLASS_CARTON: "carton",
    CLASS_PERSON: "person",
}

CLASS_IDS: Dict[str, int] = {
    "carton": CLASS_CARTON,
    "person": CLASS_PERSON,
}


class InvalidClassContractError(ValueError):
    """Raised when a detection or annotation violates the two-class contract."""
    pass


@dataclass(frozen=True)
class DetectionBox:
    """Validated 2-class bounding box detection."""

    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.class_id not in (CLASS_CARTON, CLASS_PERSON):
            raise InvalidClassContractError(
                f"Invalid class_id {self.class_id}. Contract strictly permits 0 (carton) and 1 (person)."
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise InvalidClassContractError(
                f"Confidence must be within [0.0, 1.0], got {self.confidence}"
            )
        x1, y1, x2, y2 = self.bbox
        if x2 < x1 or y2 < y1:
            raise InvalidClassContractError(
                f"Degenerate bounding box: ({x1}, {y1}, {x2}, {y2})"
            )

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]

    @property
    def is_carton(self) -> bool:
        return self.class_id == CLASS_CARTON

    @property
    def is_person(self) -> bool:
        return self.class_id == CLASS_PERSON


def is_carton(class_id: int) -> bool:
    """Return True if class_id corresponds to carton (0)."""
    return class_id == CLASS_CARTON


def is_person(class_id: int) -> bool:
    """Return True if class_id corresponds to person (1)."""
    return class_id == CLASS_PERSON


def validate_carton_event_candidate(detection: DetectionBox) -> bool:
    """
    Guard rule ensuring person detections can NEVER enter the carton event stream.
    Returns True if valid carton, raises InvalidClassContractError if person.
    """
    if detection.is_person:
        raise InvalidClassContractError(
            f"VIOLATION: Person detection (track_id={detection.track_id}) attempted to enter carton event pipeline. "
            "Person class is restricted strictly to occlusion-mask generation."
        )
    return detection.is_carton


def partition_detections(
    detections: Sequence[DetectionBox],
) -> Tuple[List[DetectionBox], List[DetectionBox]]:
    """Partition a list of detections into (cartons, persons)."""
    cartons: List[DetectionBox] = []
    persons: List[DetectionBox] = []
    for d in detections:
        if d.is_carton:
            cartons.append(d)
        elif d.is_person:
            persons.append(d)
        else:
            raise InvalidClassContractError(f"Unsupported class ID: {d.class_id}")
    return cartons, persons
