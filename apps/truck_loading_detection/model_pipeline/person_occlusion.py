"""
Person Occlusion Mask & Camera Visibility Engine (Phase 1.5).

Implements Section 2.3 specification:
  BlockedRatio_c(t) = Area(Zone B_c ∩ ∪ BBox_person) / Area(Zone B_c)
  V_c(t) = max(0.0, 1.0 - 1.5 * BlockedRatio_c(t))

Ensures person detections are used strictly as an occluder mask to compute
camera zone visibility, and NEVER trigger carton crossing events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .class_contract import CLASS_PERSON, DetectionBox, InvalidClassContractError


@dataclass(frozen=True)
class CorridorZone:
    """Rectangular corridor region in pixel coordinates."""

    name: str  # e.g. "Zone B" (Gate Core)
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"Invalid corridor zone bounds: ({self.x1}, {self.y1}, {self.x2}, {self.y2})")

    @property
    def area(self) -> float:
        return float((self.x2 - self.x1) * (self.y2 - self.y1))

    def intersection_area(self, box: Tuple[float, float, float, float]) -> float:
        """Compute intersection area with a bounding box (bx1, by1, bx2, by2)."""
        bx1, by1, bx2, by2 = box
        ix1 = max(self.x1, bx1)
        iy1 = max(self.y1, by1)
        ix2 = min(self.x2, bx2)
        iy2 = min(self.y2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        return float(iw * ih)


@dataclass
class OcclusionResult:
    """Detailed occlusion and camera visibility evaluation."""

    camera_id: str
    zone_name: str
    blocked_ratio: float  # [0.0, 1.0]
    visibility_score: float  # V_c in [0.0, 1.0]
    num_persons_detected: int
    num_persons_in_zone: int
    is_blocked: bool  # True if V_c < 0.35 (abstention threshold)

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "zone_name": self.zone_name,
            "blocked_ratio": round(self.blocked_ratio, 4),
            "visibility_score": round(self.visibility_score, 4),
            "num_persons_detected": self.num_persons_detected,
            "num_persons_in_zone": self.num_persons_in_zone,
            "is_blocked": self.is_blocked,
        }


class PersonOccluderMask:
    """
    Computes doorway corridor occlusion from detected person boxes.
    """

    def __init__(
        self,
        camera_id: str,
        zone_b: CorridorZone,
        min_person_confidence: float = 0.25,
        blocked_abstain_threshold: float = 0.35,
    ) -> None:
        self.camera_id = camera_id
        self.zone_b = zone_b
        self.min_confidence = min_person_confidence
        self.abstain_threshold = blocked_abstain_threshold

    def compute_visibility(
        self,
        person_detections: Sequence[DetectionBox],
        image_shape: Tuple[int, int] = (720, 1280),
    ) -> OcclusionResult:
        """
        Evaluate person occluder mask on Zone B.
        person_detections: list of DetectionBox with class_id == 1 (person).
        """
        # Validate that all detections are strictly person class
        valid_persons: List[DetectionBox] = []
        for d in person_detections:
            if not d.is_person:
                raise InvalidClassContractError(
                    f"Expected person detection (class_id=1), got class_id={d.class_id}"
                )
            if d.confidence >= self.min_confidence:
                valid_persons.append(d)

        if not valid_persons:
            # Zero persons -> 100% visible
            return OcclusionResult(
                camera_id=self.camera_id,
                zone_name=self.zone_b.name,
                blocked_ratio=0.0,
                visibility_score=1.0,
                num_persons_detected=0,
                num_persons_in_zone=0,
                is_blocked=False,
            )

        h, w = image_shape[:2]
        # Create binary mask for union of person boxes overlapping Zone B
        mask = np.zeros((h, w), dtype=np.uint8)

        persons_in_zone = 0
        zx1, zy1, zx2, zy2 = int(self.zone_b.x1), int(self.zone_b.y1), int(self.zone_b.x2), int(self.zone_b.y2)

        for p in valid_persons:
            px1, py1, px2, py2 = p.bbox
            if self.zone_b.intersection_area(p.bbox) > 0.0:
                persons_in_zone += 1
                # Clip to image bounds
                bx1 = max(0, min(w - 1, int(round(px1))))
                by1 = max(0, min(h - 1, int(round(py1))))
                bx2 = max(0, min(w, int(round(px2))))
                by2 = max(0, min(h, int(round(py2))))
                mask[by1:by2, bx1:bx2] = 1

        # Calculate blocked area within Zone B only
        zone_mask = mask[zy1:zy2, zx1:zx2]
        blocked_pixels = float(np.sum(zone_mask))
        total_zone_pixels = float((zy2 - zy1) * (zx2 - zx1))

        if total_zone_pixels <= 0.0:
            blocked_ratio = 0.0
        else:
            blocked_ratio = float(blocked_pixels / total_zone_pixels)
        blocked_ratio = float(np.clip(blocked_ratio, 0.0, 1.0))

        # Formula: V_c(t) = max(0, 1 - 1.5 * BlockedRatio_c(t))
        v_c = max(0.0, 1.0 - 1.5 * blocked_ratio)
        is_blocked = v_c < self.abstain_threshold

        return OcclusionResult(
            camera_id=self.camera_id,
            zone_name=self.zone_b.name,
            blocked_ratio=blocked_ratio,
            visibility_score=v_c,
            num_persons_detected=len(valid_persons),
            num_persons_in_zone=persons_in_zone,
            is_blocked=is_blocked,
        )
