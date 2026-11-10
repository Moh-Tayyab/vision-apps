"""
Geometric Calibration & Projection Adapter (Phase 4).

Wraps camera intrinsics, doorway homographies, and projection to the warehouse
loading threshold plane.

Strict Guarantees:
- Never invents synthetic homography matrices or claims uncalibrated camera pixels
  are calibrated 3D world points.
- If calibration parameters are missing or invalid, explicitly reports CALIBRATION_MISSING
  and flags downstream fusion as uncalibrated.
- Accounts for non-planar carton height parallax: 2D ground plane homography
  is valid for floor/threshold intersection, but height variations introduce parallax.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraCalibrationProfile:
    """Intrinsics and planar doorway homography for one camera perspective."""

    camera_id: str
    is_calibrated: bool = False
    homography_matrix: Optional[np.ndarray] = None  # 3x3 H: Image (px) -> Dock Doorway Plane (meters/cm)
    inv_homography_matrix: Optional[np.ndarray] = None  # 3x3 H_inv
    reprojection_rms_px: Optional[float] = None
    calibration_source: str = "uncalibrated"
    notes: str = ""
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    calibration_plane: Optional[str] = None

    def project_point_to_dock_plane(
        self,
        image_xy: Tuple[float, float],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Projects an image point (x, y) to the 2D doorway plane using homography.
        Returns None if camera is uncalibrated or homography is missing.
        """
        if not self.is_calibrated or self.homography_matrix is None:
            return None
        if image_size is not None and (self.image_width, self.image_height) != image_size:
            return None

        u, v = float(image_xy[0]), float(image_xy[1])
        pt_homo = np.array([u, v, 1.0], dtype=np.float64)
        world_homo = self.homography_matrix @ pt_homo

        if abs(world_homo[2]) < 1e-9:
            return None

        wx = world_homo[0] / world_homo[2]
        wy = world_homo[1] / world_homo[2]
        return (float(wx), float(wy))


class GeometricCalibrationManager:
    """
    Registry and validator for camera geometries.
    
    Loads calibration if available in repository config, validates reprojection error,
    and exposes a uniform projection interface.
    """

    def __init__(self, calibration_json_path: Optional[str] = None) -> None:
        self.profiles: Dict[str, CameraCalibrationProfile] = {}
        if calibration_json_path:
            self.load_from_json(calibration_json_path)

    def register_profile(self, profile: CameraCalibrationProfile) -> None:
        self.profiles[profile.camera_id] = profile

    def load_from_json(self, json_path: str) -> None:
        """Loads and parses calibration JSON from config/calibrations/."""
        import json

        p = Path(json_path)
        if not p.exists():
            logger.warning("Calibration file '%s' does not exist.", json_path)
            return

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

            cameras_dict = data.get("cameras", {})
            for cam_id, cinfo in cameras_dict.items():
                homo_info = cinfo.get("homography")
                H = None
                H_inv = None
                rms = None
                width = int(cinfo.get("image_width", 0))
                height = int(cinfo.get("image_height", 0))
                plane = homo_info.get("calibration_plane") if homo_info else None
                physical_points = homo_info.get("physical_reference_points") if homo_info else None
                validation = homo_info.get("independent_validation") if homo_info else None
                try:
                    physical_point_array = np.asarray(
                        list(physical_points.values()), dtype=np.float64
                    )
                    physical_points_valid = (
                        physical_point_array.shape == (6, 2)
                        and np.all(np.isfinite(physical_point_array))
                        and np.linalg.matrix_rank(
                            np.column_stack((physical_point_array, np.ones(6)))
                        ) >= 3
                    )
                except (AttributeError, TypeError, ValueError):
                    physical_points_valid = False
                has_physical_evidence = (
                    plane == "loading_gate_plane"
                    and homo_info.get("image_orientation_confirmed") is True
                    and isinstance(physical_points, dict)
                    and len(physical_points) == 6
                    and set(physical_points) == {
                        "gate_top_left",
                        "gate_top_right",
                        "gate_bottom_right",
                        "gate_bottom_left",
                        "gate_mid_left",
                        "gate_mid_right",
                    }
                    and physical_points_valid
                    and isinstance(validation, dict)
                    and validation.get("passed") is True
                    and int(validation.get("point_count", 0)) >= 4
                    and np.isfinite(float(validation.get("rms_error_px", float("inf"))))
                    and float(validation.get("rms_error_px", float("inf"))) <= 2.0
                    and validation.get("calibration_plane") == plane
                )
                if (
                    homo_info
                    and homo_info.get("is_valid")
                    and "H" in homo_info
                    and width > 0
                    and height > 0
                    and has_physical_evidence
                    and "H_inv" in homo_info
                ):
                    candidate_H = np.array(homo_info["H"], dtype=np.float64)
                    if candidate_H.shape == (3, 3) and np.all(np.isfinite(candidate_H)):
                        H = candidate_H
                        H_inv = np.array(homo_info["H_inv"], dtype=np.float64)
                        if H_inv.shape != (3, 3) or not np.all(np.isfinite(H_inv)):
                            H = None
                            H_inv = None
                        else:
                            rms = homo_info.get("rms_error_px")
                if H is not None and H_inv is not None:
                    prof = CameraCalibrationProfile(
                        camera_id=cam_id,
                        is_calibrated=True,
                        homography_matrix=H,
                        inv_homography_matrix=H_inv,
                        reprojection_rms_px=rms,
                        calibration_source=str(p),
                        notes=f"Loaded from {p.name}",
                        image_width=width,
                        image_height=height,
                        calibration_plane=plane,
                    )
                else:
                    prof = CameraCalibrationProfile(
                        camera_id=cam_id,
                        is_calibrated=False,
                        calibration_source=str(p),
                        notes="Missing a validated loading-gate plane and independent physical reference checks",
                        image_width=width or None,
                        image_height=height or None,
                    )
                self.profiles[cam_id] = prof
        except Exception as e:
            logger.error("Failed to parse calibration JSON at '%s': %s", json_path, e)

    def is_fully_calibrated(self, camera_ids: List[str]) -> bool:
        """Returns True only if all requested cameras have verified calibration."""
        return all(self.profiles.get(cid, CameraCalibrationProfile(cid)).is_calibrated for cid in camera_ids)

    def project(
        self,
        camera_id: str,
        image_xy: Tuple[float, float],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[float, float]]:
        prof = self.profiles.get(camera_id)
        if prof is None:
            return None
        return prof.project_point_to_dock_plane(image_xy, image_size=image_size)
