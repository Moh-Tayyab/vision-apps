"""
Doorway Homography & Landmark Estimation Engine (Phase 0.7).

Estimates planar homography H_gate from 6 physical landmarks using cv2.RANSAC
and enforces the strict undistorted RMS reprojection error <= 2.0 px requirement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class HomographyValidationError(Exception):
    """Raised when landmark geometry or homography estimation fails validation."""
    pass


REQUIRED_LANDMARK_NAMES: Tuple[str, ...] = (
    "door_top_left",
    "door_top_right",
    "door_bottom_right",
    "door_bottom_left",
    "approach_floor",
    "commit_floor",
)

# Canonical 2D world coordinates on the doorway corridor plane:
# X axis: corridor progression (-0.5m Approach, 0.0m Doorway, +0.5m Truck commit)
# Y axis: doorway vertical / lateral span (0.0 floor, 2.4m top)
DEFAULT_CANONICAL_WORLD_COORDS: Dict[str, Tuple[float, float]] = {
    "door_top_left": (-1.20, 2.40),
    "door_top_right": (1.20, 2.40),
    "door_bottom_right": (1.20, 0.00),
    "door_bottom_left": (-1.20, 0.00),
    "approach_floor": (0.00, -0.50),  # 0.5m before doorway along travel corridor
    "commit_floor": (0.00, 0.50),     # 0.5m inside truck doorway
}


@dataclass
class LandmarkSet:
    """Set of 6 annotated physical landmarks."""

    points: Dict[str, Tuple[float, float]]  # landmark_name -> (u, v) pixels
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        missing = [name for name in REQUIRED_LANDMARK_NAMES if name not in self.points]
        if missing:
            raise HomographyValidationError(
                f"Missing required landmarks: {missing}. Exactly 6 landmarks required: {REQUIRED_LANDMARK_NAMES}"
            )

        for name, (u, v) in self.points.items():
            if not (math.isfinite(u) and math.isfinite(v)):
                raise HomographyValidationError(f"Landmark '{name}' has non-finite coordinates ({u}, {v})")
            if not (-50 <= u <= self.image_width + 50) or not (-50 <= v <= self.image_height + 50):
                raise HomographyValidationError(
                    f"Landmark '{name}' coordinate ({u}, {v}) is unreasonably far from image bounds ({self.image_width}x{self.image_height})"
                )

    def to_array(self) -> np.ndarray:
        """Return (6, 2) array of coordinates in standardized landmark order."""
        return np.array([self.points[name] for name in REQUIRED_LANDMARK_NAMES], dtype=np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        return {name: [float(u), float(v)] for name, (u, v) in self.points.items()}


@dataclass
class HomographyResult:
    """Validated homography estimation result."""

    H: np.ndarray  # (3, 3) image -> world homography
    H_inv: np.ndarray  # (3, 3) world -> image inverse homography
    rms_error_px: float
    max_error_px: float
    per_point_errors_px: Dict[str, float]
    inlier_count: int
    inlier_ratio: float
    is_valid: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "H": self.H.tolist(),
            "H_inv": self.H_inv.tolist(),
            "rms_error_px": float(self.rms_error_px),
            "max_error_px": float(self.max_error_px),
            "per_point_errors_px": {k: float(v) for k, v in self.per_point_errors_px.items()},
            "inlier_count": int(self.inlier_count),
            "inlier_ratio": float(self.inlier_ratio),
            "is_valid": bool(self.is_valid),
            "rejection_reason": self.rejection_reason,
        }


class DoorwayHomographyEstimator:
    """Estimates and validates doorway plane homography using RANSAC."""

    def __init__(
        self,
        canonical_world_coords: Optional[Dict[str, Tuple[float, float]]] = None,
        max_reprojection_rms: float = 2.0,  # Strict acceptance requirement: RMS <= 2.0 px
        ransac_threshold_px: float = 3.0,
    ) -> None:
        self.world_coords = canonical_world_coords or DEFAULT_CANONICAL_WORLD_COORDS
        self.max_rms = max_reprojection_rms
        self.ransac_threshold = ransac_threshold_px

        # Standard destination points array matching REQUIRED_LANDMARK_NAMES order
        self.dst_world_pts = np.array(
            [self.world_coords[name] for name in REQUIRED_LANDMARK_NAMES], dtype=np.float32
        )

    def estimate(self, landmarks: LandmarkSet) -> HomographyResult:
        """
        Estimate H mapping image pixel coordinates (u, v) to world coordinates (X, Y).
        Computes forward and backward reprojection errors and validates RMS <= 2.0 px.
        """
        src_img_pts = landmarks.to_array()

        # Find homography: src_img_pts -> dst_world_pts
        H, mask = cv2.findHomography(
            src_img_pts, self.dst_world_pts, cv2.RANSAC, self.ransac_threshold
        )

        if H is None:
            return HomographyResult(
                H=np.eye(3),
                H_inv=np.eye(3),
                rms_error_px=float("inf"),
                max_error_px=float("inf"),
                per_point_errors_px={},
                inlier_count=0,
                inlier_ratio=0.0,
                is_valid=False,
                rejection_reason="cv2.findHomography returned None (degenerate point set)",
            )

        # Check invertibility and numerical stability
        det = np.linalg.det(H)
        if abs(det) < 1e-8 or not np.isfinite(det):
            return HomographyResult(
                H=H,
                H_inv=np.eye(3),
                rms_error_px=float("inf"),
                max_error_px=float("inf"),
                per_point_errors_px={},
                inlier_count=0,
                inlier_ratio=0.0,
                is_valid=False,
                rejection_reason=f"Singular or near-singular homography (det={det:.2e})",
            )

        H_inv = np.linalg.inv(H)

        # Inlier statistics
        inliers = mask.flatten() if mask is not None else np.ones(len(src_img_pts), dtype=int)
        inlier_count = int(np.sum(inliers))
        inlier_ratio = float(inlier_count / len(src_img_pts))

        # Reprojection error in PIXEL space:
        # Reproject world points back to image plane using H_inv
        pts_world_homo = np.hstack([self.dst_world_pts, np.ones((len(self.dst_world_pts), 1))])
        reprojected_homo = (H_inv @ pts_world_homo.T).T
        reprojected_px = reprojected_homo[:, :2] / reprojected_homo[:, 2:3]

        diff_px = src_img_pts - reprojected_px
        sq_errors = np.sum(diff_px ** 2, axis=1)
        point_errors = np.sqrt(sq_errors)

        per_point_errors: Dict[str, float] = {}
        for idx, name in enumerate(REQUIRED_LANDMARK_NAMES):
            per_point_errors[name] = float(point_errors[idx])

        rms_error_px = float(np.sqrt(np.mean(sq_errors)))
        max_error_px = float(np.max(point_errors))

        # Enforce validation criteria
        rejection_reason: Optional[str] = None
        is_valid = True

        if inlier_count < 4:
            is_valid = False
            rejection_reason = f"Insufficient RANSAC inliers: {inlier_count} / {len(src_img_pts)}"
        elif rms_error_px > self.max_rms:
            is_valid = False
            rejection_reason = (
                f"Reprojection RMS error {rms_error_px:.3f} px exceeds strict threshold {self.max_rms:.2f} px"
            )

        return HomographyResult(
            H=H,
            H_inv=H_inv,
            rms_error_px=rms_error_px,
            max_error_px=max_error_px,
            per_point_errors_px=per_point_errors,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            is_valid=is_valid,
            rejection_reason=rejection_reason,
        )
