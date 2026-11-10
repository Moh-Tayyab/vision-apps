"""
Calibration Persistence Engine (Phase 0.3).

Provides atomic serialization and deserialization of dock camera calibrations
at config/calibrations/{site_id}_{dock_id}_v2.json with strict schema validation.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class CalibrationStorageError(Exception):
    """Raised when calibration persistence or schema validation fails."""
    pass


@dataclass
class CameraCalibrationData:
    """Per-camera calibration and commissioning record."""

    camera_id: str
    camera_model: str
    image_width: int
    image_height: int
    K: np.ndarray  # (3, 3) float64
    D: np.ndarray  # (5,) or (4,) float64
    rms_error: float
    focal_length_mm: Optional[float] = None
    per_view_errors: List[float] = field(default_factory=list)
    landmarks: Optional[Dict[str, List[float]]] = None
    homography: Optional[Dict[str, Any]] = None
    watchdog_baseline: Optional[Dict[str, Any]] = None
    alignment_status: str = "CALIBRATED_OPERATIONAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "camera_model": self.camera_model,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "focal_length_mm": float(self.focal_length_mm) if self.focal_length_mm is not None else None,
            "K": self.K.tolist() if isinstance(self.K, np.ndarray) else self.K,
            "D": self.D.flatten().tolist() if isinstance(self.D, np.ndarray) else self.D,
            "rms_error": float(self.rms_error),
            "per_view_errors": [float(e) for e in self.per_view_errors],
            "landmarks": self.landmarks,
            "homography": self.homography,
            "watchdog_baseline": self.watchdog_baseline,
            "alignment_status": self.alignment_status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraCalibrationData":
        if not isinstance(data, dict):
            raise CalibrationStorageError(f"Expected dict for camera data, got {type(data)}")

        cam_id = data.get("camera_id")
        if not cam_id:
            raise CalibrationStorageError("Missing 'camera_id' in calibration data")

        raw_k = data.get("K")
        if raw_k is None:
            raise CalibrationStorageError(f"Missing intrinsic matrix K for camera {cam_id}")
        try:
            k_arr = np.array(raw_k, dtype=np.float64)
            if k_arr.shape != (3, 3) or not np.all(np.isfinite(k_arr)):
                raise ValueError("K must be a 3x3 finite numeric matrix")
        except Exception as e:
            raise CalibrationStorageError(f"Invalid K matrix in camera {cam_id}: {e}") from e

        raw_d = data.get("D")
        if raw_d is None:
            raise CalibrationStorageError(f"Missing distortion vector D for camera {cam_id}")
        try:
            d_arr = np.array(raw_d, dtype=np.float64).flatten()
            if not np.all(np.isfinite(d_arr)):
                raise ValueError("D must contain finite numeric values")
        except Exception as e:
            raise CalibrationStorageError(f"Invalid D vector in camera {cam_id}: {e}") from e

        return cls(
            camera_id=str(cam_id),
            camera_model=str(data.get("camera_model", "pinhole")),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            K=k_arr,
            D=d_arr,
            rms_error=float(data.get("rms_error", 0.0)),
            focal_length_mm=float(data["focal_length_mm"]) if data.get("focal_length_mm") is not None else None,
            per_view_errors=data.get("per_view_errors", []),
            landmarks=data.get("landmarks"),
            homography=data.get("homography"),
            watchdog_baseline=data.get("watchdog_baseline"),
            alignment_status=str(data.get("alignment_status", "CALIBRATED_OPERATIONAL")),
        )


@dataclass
class DockCalibration:
    """Dock-level calibration containing all participating cameras."""

    site_id: str
    dock_id: str
    schema_version: str = "2.0"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    opencv_version: str = field(
        default_factory=lambda: __import__("cv2").__version__
    )
    cameras: Dict[str, CameraCalibrationData] = field(default_factory=dict)

    def add_camera(self, cam_data: CameraCalibrationData) -> None:
        self.cameras[cam_data.camera_id] = cam_data

    def get_camera(self, camera_id: str) -> CameraCalibrationData:
        if camera_id not in self.cameras:
            raise KeyError(f"Camera '{camera_id}' not found in dock {self.dock_id}")
        return self.cameras[camera_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "site_id": self.site_id,
            "dock_id": self.dock_id,
            "created_at": self.created_at,
            "opencv_version": self.opencv_version,
            "cameras": {cid: c.to_dict() for cid, c in self.cameras.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DockCalibration":
        if not isinstance(data, dict):
            raise CalibrationStorageError("Calibration file root must be a JSON object")

        version = data.get("schema_version")
        if version != "2.0":
            raise CalibrationStorageError(
                f"Unsupported schema_version '{version}', expected '2.0'"
            )

        site_id = data.get("site_id")
        dock_id = data.get("dock_id")
        if not site_id or not dock_id:
            raise CalibrationStorageError("Missing site_id or dock_id in calibration record")

        cams_dict: Dict[str, CameraCalibrationData] = {}
        raw_cams = data.get("cameras", {})
        if not isinstance(raw_cams, dict):
            raise CalibrationStorageError("'cameras' field must be a dictionary")

        for cid, cdata in raw_cams.items():
            if not isinstance(cdata, dict):
                raise CalibrationStorageError(f"Camera record for '{cid}' must be a dict")
            cdata["camera_id"] = cid
            cams_dict[cid] = CameraCalibrationData.from_dict(cdata)

        return cls(
            site_id=str(site_id),
            dock_id=str(dock_id),
            schema_version=str(version),
            created_at=str(data.get("created_at", "")),
            opencv_version=str(data.get("opencv_version", "")),
            cameras=cams_dict,
        )


class CalibrationStorage:
    """Manages file storage and atomic reads/writes of dock calibrations."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None) -> None:
        if base_dir is None:
            # Default to config/calibrations in repo
            self.base_dir = Path("config/calibrations")
        else:
            self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, site_id: str, dock_id: str) -> Path:
        """Construct canonical file path: {site_id}_{dock_id}_v2.json"""
        safe_site = site_id.strip().replace("/", "_").replace("\\", "_")
        safe_dock = dock_id.strip().replace("/", "_").replace("\\", "_")
        return self.base_dir / f"{safe_site}_{safe_dock}_v2.json"

    def exists(self, site_id: str, dock_id: str) -> bool:
        return self.get_path(site_id, dock_id).exists()

    def load(self, site_id: str, dock_id: str) -> DockCalibration:
        """Load and validate calibration JSON."""
        target_path = self.get_path(site_id, dock_id)
        if not target_path.exists():
            raise FileNotFoundError(f"Calibration file not found at: {target_path}")

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise CalibrationStorageError(f"Malformed JSON in {target_path}: {e}") from e
        except Exception as e:
            raise CalibrationStorageError(f"Error reading {target_path}: {e}") from e

        # Validate that site_id and dock_id match
        if raw_data.get("site_id") != site_id or raw_data.get("dock_id") != dock_id:
            raise CalibrationStorageError(
                f"File content site/dock mismatch: requested ({site_id}, {dock_id}), "
                f"found ({raw_data.get('site_id')}, {raw_data.get('dock_id')})"
            )

        return DockCalibration.from_dict(raw_data)

    def save(self, calibration: DockCalibration) -> Path:
        """
        Atomically save DockCalibration to disk.
        Writes to a temporary file in the same directory, then renames.
        """
        target_path = self.get_path(calibration.site_id, calibration.dock_id)
        data = calibration.to_dict()

        # Strict finite float verification on entire dict
        def _check_finite(obj: Any) -> None:
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    raise CalibrationStorageError(f"Non-finite float detected: {obj}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _check_finite(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    _check_finite(v)

        _check_finite(data)

        json_bytes = json.dumps(data, indent=2).encode("utf-8")

        # Atomic write: create temp file in self.base_dir to guarantee same filesystem for atomic rename
        temp_file = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(self.base_dir),
            delete=False,
            prefix="calib_tmp_",
            suffix=".json",
        )
        try:
            temp_file.write(json_bytes)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            # Atomic replacement
            os.replace(temp_file.name, target_path)
        except Exception as e:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
            raise CalibrationStorageError(f"Failed to atomically write calibration to {target_path}: {e}") from e

        return target_path
