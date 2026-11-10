"""
Undistortion and Rectification LUT Engine (Phase 0.4 & 0.5).

Precomputes CV_16SC2 / uint16 fixed-point lookup tables at startup to guarantee
zero runtime heap allocation during remap, and provides geometrically consistent
point undistortion via cv2.undistortPoints(..., P=K_new).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np


class UndistortionError(Exception):
    """Raised when undistortion map precomputation or execution fails."""
    pass


@dataclass
class CachedRemapLUT:
    """Precomputed remap lookup tables and associated metadata."""

    camera_id: str
    image_width: int
    image_height: int
    map1: np.ndarray  # int16 (H, W, 2) for CV_16SC2, or float32 (H, W) for CV_32FC1
    map2: np.ndarray  # uint16 (H, W) for CV_16SC2, or float32 (H, W) for CV_32FC1
    map_type: int
    K: np.ndarray
    D: np.ndarray
    K_new: np.ndarray
    cache_key: str
    created_at_mono: float = field(default_factory=time.monotonic)


class UndistortionLUTManager:
    """
    Manages precomputation, caching, and execution of camera undistortion maps.

    Technical Note on CV_16SC2 vs 'float16':
    cv2.initUndistortRectifyMap(..., m1type=cv2.CV_16SC2) generates:
      - map1: int16 array of shape (H, W, 2) storing integer pixel offsets.
      - map2: uint16 array of shape (H, W) storing subpixel interpolation weights.
    This is NOT IEEE 754 float16. In OpenCV, CV_16SC2 + uint16 is the optimized
    SIMD fixed-point format that executes faster than floating-point remapping.
    """

    def __init__(self, default_map_type: int = cv2.CV_16SC2) -> None:
        self.default_map_type = default_map_type
        self._cache: Dict[str, CachedRemapLUT] = {}

    @staticmethod
    def compute_cache_key(
        camera_id: str, width: int, height: int, K: np.ndarray, D: np.ndarray, K_new: np.ndarray
    ) -> str:
        """Generate unique hash representing the calibration and resolution state."""
        h = hashlib.sha256()
        h.update(camera_id.encode("utf-8"))
        h.update(f":{width}x{height}:".encode("utf-8"))
        h.update(K.tobytes())
        h.update(D.tobytes())
        h.update(K_new.tobytes())
        return h.hexdigest()

    def get_or_create_lut(
        self,
        camera_id: str,
        image_size: Tuple[int, int],
        K: np.ndarray,
        D: np.ndarray,
        K_new: Optional[np.ndarray] = None,
        camera_model: str = "pinhole",
        map_type: Optional[int] = None,
    ) -> CachedRemapLUT:
        """
        Retrieve cached LUT or precompute a new one if parameters or resolution changed.
        image_size: (width, height)
        """
        w, h = image_size
        if w <= 0 or h <= 0:
            raise UndistortionError(f"Invalid image size {(w, h)}")

        # By default, use K as K_new (maintains original scale and focal length)
        if K_new is None:
            K_new = K.copy()

        m_type = map_type if map_type is not None else self.default_map_type
        cache_key = self.compute_cache_key(camera_id, w, h, K, D, K_new)

        # Check existing cache
        existing = self._cache.get(camera_id)
        if existing is not None and existing.cache_key == cache_key:
            return existing

        # Precompute new lookup tables
        try:
            if camera_model in ("pinhole", "radtan"):
                map1, map2 = cv2.initUndistortRectifyMap(
                    K, D, None, K_new, (w, h), m_type
                )
            elif camera_model == "fisheye":
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), K_new, (w, h), m_type
                )
            else:
                raise UndistortionError(f"Unsupported camera model '{camera_model}'")
        except Exception as e:
            raise UndistortionError(f"Failed to generate undistortion LUT: {e}") from e

        cached = CachedRemapLUT(
            camera_id=camera_id,
            image_width=w,
            image_height=h,
            map1=map1,
            map2=map2,
            map_type=m_type,
            K=K,
            D=D,
            K_new=K_new,
            cache_key=cache_key,
        )
        self._cache[camera_id] = cached
        return cached

    def remap_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        K: np.ndarray,
        D: np.ndarray,
        K_new: Optional[np.ndarray] = None,
        camera_model: str = "pinhole",
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """Remap a frame using cached lookup tables."""
        h, w = frame.shape[:2]
        lut = self.get_or_create_lut(
            camera_id=camera_id,
            image_size=(w, h),
            K=K,
            D=D,
            K_new=K_new,
            camera_model=camera_model,
        )
        return cv2.remap(frame, lut.map1, lut.map2, interpolation)

    def invalidate(self, camera_id: str) -> None:
        """Remove cached LUT for a camera."""
        self._cache.pop(camera_id, None)

    def clear(self) -> None:
        """Clear all cached LUTs."""
        self._cache.clear()


class GeometricUndistorter:
    """
    Unified geometric undistortion service guaranteeing consistency between
    full-frame cv2.remap and point cv2.undistortPoints.
    """

    def __init__(
        self,
        camera_id: str,
        image_size: Tuple[int, int],
        K: np.ndarray,
        D: np.ndarray,
        K_new: Optional[np.ndarray] = None,
        camera_model: str = "pinhole",
        lut_manager: Optional[UndistortionLUTManager] = None,
    ) -> None:
        self.camera_id = camera_id
        self.image_size = image_size
        self.K = np.array(K, dtype=np.float64)
        self.D = np.array(D, dtype=np.float64).flatten()
        self.K_new = np.array(K_new if K_new is not None else K, dtype=np.float64)
        self.camera_model = camera_model
        self.lut_manager = lut_manager or UndistortionLUTManager()

        # Warm up LUT precomputation at initialization
        self.lut = self.lut_manager.get_or_create_lut(
            camera_id=self.camera_id,
            image_size=self.image_size,
            K=self.K,
            D=self.D,
            K_new=self.K_new,
            camera_model=self.camera_model,
        )

    def undistort_image(
        self, frame: np.ndarray, interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """Undistort full camera frame using precomputed LUT."""
        return cv2.remap(frame, self.lut.map1, self.lut.map2, interpolation)

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Undistort 2D pixel coordinates (N, 2) or (N, 1, 2) into the EXACT SAME
        pixel coordinate system as the remapped image by providing P=K_new.

        CRITICAL GEOMETRIC GUARANTEE:
        Providing P=self.K_new ensures the output coordinates match the rectified
        pixel grid. If P were omitted, OpenCV would return normalized ray coordinates.
        """
        pts = np.asarray(points, dtype=np.float32)
        orig_shape = pts.shape

        if pts.ndim == 2 and pts.shape[1] == 2:
            pts_reshaped = pts.reshape(-1, 1, 2)
        elif pts.ndim == 3 and pts.shape[2] == 2:
            pts_reshaped = pts
        else:
            raise ValueError(f"Points must have shape (N, 2) or (N, 1, 2), got {pts.shape}")

        if self.camera_model in ("pinhole", "radtan"):
            undistorted = cv2.undistortPoints(
                pts_reshaped, self.K, self.D, P=self.K_new
            )
        elif self.camera_model == "fisheye":
            # cv2.fisheye.undistortPoints requires float64 and R=eye(3)
            undistorted = cv2.fisheye.undistortPoints(
                pts_reshaped.astype(np.float64),
                self.K,
                self.D,
                R=np.eye(3),
                P=self.K_new,
            )
        else:
            raise UndistortionError(f"Unsupported camera model: {self.camera_model}")

        return undistorted.reshape(orig_shape)
