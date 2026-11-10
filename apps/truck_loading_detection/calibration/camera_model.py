"""
Camera Model Configuration Schema and Validation.

Specifies and validates intrinsic matrix K, distortion coefficients D,
camera model type, focal length, and sensor resolution.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml


class CameraModelValidationError(ValueError):
    """Raised when camera model configuration violates the schema."""
    pass


@dataclass(frozen=True)
class CameraModel:
    """Validated camera model specification."""

    model_id: str
    camera_model: str  # "pinhole" or "fisheye"
    image_width: int
    image_height: int
    focal_length_mm: float
    K: np.ndarray  # (3, 3) float64
    D: np.ndarray  # (5,) for pinhole or (4,) for fisheye, float64
    description: str = ""
    pixel_size_um: Optional[float] = None

    def __post_init__(self) -> None:
        # Validate model type
        if self.camera_model not in ("pinhole", "radtan", "fisheye"):
            raise CameraModelValidationError(
                f"Invalid camera_model '{self.camera_model}'. Must be 'pinhole' or 'fisheye'."
            )

        # Validate image dimensions
        if not (isinstance(self.image_width, (int, np.integer)) and self.image_width > 0):
            raise CameraModelValidationError(
                f"image_width must be a positive integer, got {self.image_width}"
            )
        if not (isinstance(self.image_height, (int, np.integer)) and self.image_height > 0):
            raise CameraModelValidationError(
                f"image_height must be a positive integer, got {self.image_height}"
            )

        # Validate focal length
        if not (isinstance(self.focal_length_mm, (int, float, np.floating)) and self.focal_length_mm > 0):
            raise CameraModelValidationError(
                f"focal_length_mm must be a positive number, got {self.focal_length_mm}"
            )
        if math.isnan(self.focal_length_mm) or math.isinf(self.focal_length_mm):
            raise CameraModelValidationError(
                "focal_length_mm cannot be NaN or infinite"
            )

        # Validate K matrix
        if not isinstance(self.K, np.ndarray) or self.K.shape != (3, 3):
            raise CameraModelValidationError(
                f"K matrix must be a 3x3 numpy array, got shape {getattr(self.K, 'shape', None)}"
            )
        if not np.all(np.isfinite(self.K)):
            raise CameraModelValidationError(
                "K matrix contains non-finite values (NaN or Inf)"
            )

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        if fx <= 0 or fy <= 0:
            raise CameraModelValidationError(
                f"Focal lengths (fx, fy) must be positive, got fx={fx}, fy={fy}"
            )
        if not (0 <= cx <= self.image_width):
            raise CameraModelValidationError(
                f"Principal point cx={cx} outside image bounds [0, {self.image_width}]"
            )
        if not (0 <= cy <= self.image_height):
            raise CameraModelValidationError(
                f"Principal point cy={cy} outside image bounds [0, {self.image_height}]"
            )

        # Validate D vector
        if not isinstance(self.D, np.ndarray):
            raise CameraModelValidationError(
                f"D distortion coefficients must be a numpy array, got {type(self.D)}"
            )
        if not np.all(np.isfinite(self.D)):
            raise CameraModelValidationError(
                "D distortion vector contains non-finite values (NaN or Inf)"
            )

        d_flat = self.D.flatten()
        if self.camera_model in ("pinhole", "radtan"):
            if d_flat.size not in (4, 5, 8):
                raise CameraModelValidationError(
                    f"Pinhole distortion vector D must have 5 coefficients [k1, k2, p1, p2, k3] "
                    f"(or 4/8), got {d_flat.size}"
                )
        elif self.camera_model == "fisheye":
            if d_flat.size != 4:
                raise CameraModelValidationError(
                    f"Fisheye distortion vector D must have exactly 4 coefficients [k1, k2, k3, k4], got {d_flat.size}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert camera model to standard dictionary."""
        return {
            "model_id": self.model_id,
            "description": self.description,
            "camera_model": self.camera_model,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "focal_length_mm": float(self.focal_length_mm),
            "pixel_size_um": float(self.pixel_size_um) if self.pixel_size_um is not None else None,
            "K": self.K.tolist(),
            "D": self.D.flatten().tolist(),
        }

    @classmethod
    def from_dict(cls, model_id: str, data: Dict[str, Any]) -> "CameraModel":
        """Construct and validate CameraModel from raw dictionary."""
        if not isinstance(data, dict):
            raise CameraModelValidationError(f"Expected dict for camera model '{model_id}', got {type(data)}")

        camera_model = data.get("camera_model") or data.get("model")
        if not camera_model:
            raise CameraModelValidationError(f"Missing required field 'camera_model' in model '{model_id}'")

        width = data.get("image_width") or data.get("width")
        height = data.get("image_height") or data.get("height")
        if width is None or height is None:
            raise CameraModelValidationError(f"Missing image_width or image_height in model '{model_id}'")

        focal_mm = data.get("focal_length_mm") or data.get("focal_length")
        if focal_mm is None:
            raise CameraModelValidationError(f"Missing focal_length_mm in model '{model_id}'")

        raw_k = data.get("K") or data.get("K_c")
        if raw_k is None:
            raise CameraModelValidationError(f"Missing intrinsic matrix K in model '{model_id}'")
        try:
            k_arr = np.array(raw_k, dtype=np.float64)
        except Exception as e:
            raise CameraModelValidationError(f"Invalid K matrix format in model '{model_id}': {e}") from e

        raw_d = data.get("D") or data.get("D_c")
        if raw_d is None:
            raise CameraModelValidationError(f"Missing distortion vector D in model '{model_id}'")
        try:
            d_arr = np.array(raw_d, dtype=np.float64)
        except Exception as e:
            raise CameraModelValidationError(f"Invalid D vector format in model '{model_id}': {e}") from e

        return cls(
            model_id=str(model_id),
            camera_model=str(camera_model).lower(),
            image_width=int(width),
            image_height=int(height),
            focal_length_mm=float(focal_mm),
            K=k_arr,
            D=d_arr,
            description=str(data.get("description", "")),
            pixel_size_um=float(data["pixel_size_um"]) if "pixel_size_um" in data and data["pixel_size_um"] is not None else None,
        )


class CameraModelRegistry:
    """Registry managing available camera models loaded from YAML configuration."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        self.models: Dict[str, CameraModel] = {}
        if config_path:
            self.load_yaml(config_path)

    def load_yaml(self, path: Union[str, Path]) -> None:
        """Load and validate camera models from a YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Camera models file not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = yaml.safe_load(f)

        if not isinstance(raw_content, dict):
            raise CameraModelValidationError(f"Invalid YAML content in {path}: expected mapping at root")

        models_dict = raw_content.get("models")
        if not isinstance(models_dict, dict):
            raise CameraModelValidationError(f"Missing 'models' mapping in {path}")

        errors: List[str] = []
        for model_id, model_data in models_dict.items():
            try:
                model = CameraModel.from_dict(model_id, model_data)
                self.models[model_id] = model
            except CameraModelValidationError as e:
                errors.append(f"[{model_id}]: {str(e)}")

        if errors:
            raise CameraModelValidationError(
                f"Failed to validate camera models in {path}:\n" + "\n".join(errors)
            )

    def get(self, model_id: str) -> CameraModel:
        """Retrieve camera model by ID."""
        if model_id not in self.models:
            raise KeyError(
                f"Camera model '{model_id}' not found in registry. "
                f"Available models: {list(self.models.keys())}"
            )
        return self.models[model_id]

    def register(self, model: CameraModel) -> None:
        """Register a validated camera model."""
        self.models[model.model_id] = model
