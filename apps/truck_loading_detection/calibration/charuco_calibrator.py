"""
ChArUco Board Calibration Pipeline (Phase 0.2).

Supports 9x6 ChArUco board creation, corner detection, calibration parameter
estimation (pinhole & fisheye), and comprehensive validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


class CalibrationError(Exception):
    """Raised when camera calibration fails or does not meet quality criteria."""
    pass


@dataclass(frozen=True)
class CharucoConfig:
    """ChArUco board geometry and detector configuration."""

    squares_x: int = 9  # 9 squares along X axis
    squares_y: int = 6  # 6 squares along Y axis
    square_length_m: float = 0.040  # 40 mm per square
    marker_length_m: float = 0.020  # 20 mm per marker
    dictionary_id: int = cv2.aruco.DICT_5X5_100
    min_corners_per_view: int = 6
    min_views: int = 5
    max_rms_error: float = 1.50  # maximum allowed reprojection RMS in pixels

    def create_board(self) -> cv2.aruco.CharucoBoard:
        """Instantiate OpenCV CharucoBoard with configured geometry."""
        dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        # OpenCV 4.7+ and 5.0 use size tuple (squares_x, squares_y)
        try:
            return cv2.aruco.CharucoBoard(
                (self.squares_x, self.squares_y),
                self.square_length_m,
                self.marker_length_m,
                dictionary,
            )
        except TypeError:
            # Older OpenCV 4.x signature: (squares_x, squares_y, square_len, marker_len, dict)
            return cv2.aruco.CharucoBoard_create(
                self.squares_x,
                self.squares_y,
                self.square_length_m,
                self.marker_length_m,
                dictionary,
            )


@dataclass
class CalibrationResult:
    """Complete validated calibration output."""

    camera_model: str
    image_width: int
    image_height: int
    K: np.ndarray  # (3, 3)
    D: np.ndarray  # (5,) or (4,)
    rms_error: float
    per_view_errors: List[float]
    num_views_used: int
    num_views_total: int
    opencv_version: str = field(default_factory=lambda: cv2.__version__)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_model": self.camera_model,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "K": self.K.tolist(),
            "D": self.D.flatten().tolist(),
            "rms_error": float(self.rms_error),
            "per_view_errors": [float(e) for e in self.per_view_errors],
            "num_views_used": int(self.num_views_used),
            "num_views_total": int(self.num_views_total),
            "opencv_version": self.opencv_version,
        }


class CharucoCalibrator:
    """Manages observation collection, corner detection, and calibration execution."""

    def __init__(self, config: Optional[CharucoConfig] = None) -> None:
        self.config = config or CharucoConfig()
        self.board = self.config.create_board()
        self._init_detector()

        # Observation buffers
        self.all_obj_points: List[np.ndarray] = []
        self.all_img_points: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None
        self.total_frames_processed = 0

    def _init_detector(self) -> None:
        if hasattr(cv2.aruco, "CharucoDetector"):
            self.detector = cv2.aruco.CharucoDetector(self.board)
        else:
            self.detector = None

    def detect_corners(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ChArUco corners in an image.
        Returns: (charuco_corners, charuco_ids) or (None, None).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if self.detector is not None:
            c_corners, c_ids, _, _ = self.detector.detectBoard(gray)
            if c_corners is not None and len(c_corners) > 0:
                return c_corners, c_ids
            return None, None

        # Fallback for OpenCV 4.x
        d = cv2.aruco.getPredefinedDictionary(self.config.dictionary_id)
        params = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=params)
        if ids is not None and len(ids) > 0:
            count, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board
            )
            if count > 0 and c_corners is not None:
                return c_corners, c_ids
        return None, None

    def add_view(self, image: np.ndarray) -> bool:
        """
        Process a calibration image and store observations if sufficient corners are found.
        Returns True if the view was accepted, False otherwise.
        """
        h, w = image.shape[:2]
        if self.image_size is None:
            self.image_size = (w, h)
        elif self.image_size != (w, h):
            raise CalibrationError(
                f"Image size mismatch: expected {self.image_size}, got {(w, h)}"
            )

        self.total_frames_processed += 1
        c_corners, c_ids = self.detect_corners(image)

        if c_corners is None or c_ids is None or len(c_corners) < self.config.min_corners_per_view:
            return False

        # Extract 3D object points matching the detected IDs
        all_corners_3d = self.board.getChessboardCorners()
        obj_pts = all_corners_3d[c_ids.flatten()].astype(np.float32)
        img_pts = c_corners.reshape(-1, 2).astype(np.float32)

        self.all_obj_points.append(obj_pts)
        self.all_img_points.append(img_pts)
        return True

    def add_corners(
        self,
        obj_points: np.ndarray,
        img_points: np.ndarray,
        image_size: Tuple[int, int],
    ) -> None:
        """Directly add 3D/2D point correspondences (e.g. for synthetic testing)."""
        w, h = image_size
        if self.image_size is None:
            self.image_size = (w, h)
        elif self.image_size != (w, h):
            raise CalibrationError(f"Image size mismatch: {self.image_size} vs {(w, h)}")

        if len(obj_points) < self.config.min_corners_per_view:
            return

        self.total_frames_processed += 1
        self.all_obj_points.append(obj_points.astype(np.float32))
        self.all_img_points.append(img_points.astype(np.float32))

    def calibrate(self, camera_model: str = "pinhole") -> CalibrationResult:
        """
        Execute calibration and validate the resulting camera parameters.
        camera_model: 'pinhole' or 'fisheye'
        """
        if len(self.all_obj_points) < self.config.min_views:
            raise CalibrationError(
                f"Insufficient usable calibration views: {len(self.all_obj_points)} < {self.config.min_views}"
            )
        if self.image_size is None:
            raise CalibrationError("No image dimensions established")

        w, h = self.image_size

        if camera_model == "pinhole":
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                self.all_obj_points,
                self.all_img_points,
                (w, h),
                None,
                None,
                flags=cv2.CALIB_RATIONAL_MODEL if False else 0,
            )
        elif camera_model == "fisheye":
            # cv2.fisheye requires specific point shapes: (1, N, 3) and (1, N, 2)
            obj_pts_fish = [p.reshape(1, -1, 3).astype(np.float64) for p in self.all_obj_points]
            img_pts_fish = [p.reshape(1, -1, 2).astype(np.float64) for p in self.all_img_points]
            K = np.zeros((3, 3), dtype=np.float64)
            D = np.zeros((4, 1), dtype=np.float64)
            rvecs = []
            tvecs = []
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                obj_pts_fish,
                img_pts_fish,
                (w, h),
                K,
                D,
                rvecs,
                tvecs,
                flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
            )
        else:
            raise CalibrationError(f"Unsupported camera model: '{camera_model}'")

        # Compute per-view reprojection errors
        per_view_errors: List[float] = []
        total_sq_err = 0.0
        total_points = 0

        for i in range(len(self.all_obj_points)):
            obj_pts = self.all_obj_points[i]
            img_pts = self.all_img_points[i]

            if camera_model == "pinhole":
                proj_pts, _ = cv2.projectPoints(obj_pts, rvecs[i], tvecs[i], K, D)
                proj_pts = proj_pts.reshape(-1, 2)
            else:
                proj_pts, _ = cv2.fisheye.projectPoints(
                    obj_pts.reshape(1, -1, 3), rvecs[i], tvecs[i], K, D
                )
                proj_pts = proj_pts.reshape(-1, 2)

            diff = img_pts - proj_pts
            sq_err = np.sum(diff ** 2)
            view_rms = math.sqrt(sq_err / len(obj_pts))
            per_view_errors.append(view_rms)

            total_sq_err += sq_err
            total_points += len(obj_pts)

        computed_rms = math.sqrt(total_sq_err / total_points) if total_points > 0 else float("inf")

        # Validate parameters
        self._validate_calibration(K, D, (w, h), computed_rms)

        return CalibrationResult(
            camera_model=camera_model,
            image_width=w,
            image_height=h,
            K=K,
            D=D,
            rms_error=computed_rms,
            per_view_errors=per_view_errors,
            num_views_used=len(self.all_obj_points),
            num_views_total=self.total_frames_processed,
        )

    def _validate_calibration(
        self, K: np.ndarray, D: np.ndarray, size: Tuple[int, int], rms: float
    ) -> None:
        """Validate numeric plausibility of calibrated matrices."""
        w, h = size
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(D)):
            raise CalibrationError("Calibration produced non-finite values in K or D")

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        if fx <= 0 or fy <= 0:
            raise CalibrationError(f"Focal lengths must be positive, got fx={fx}, fy={fy}")

        # Focal length should be reasonable for standard cameras (e.g. within 0.1x to 10x sensor size)
        if fx < 100 or fx > 20000 or fy < 100 or fy > 20000:
            raise CalibrationError(f"Implausible focal lengths: fx={fx}, fy={fy}")

        if not (0 <= cx <= w) or not (0 <= cy <= h):
            raise CalibrationError(
                f"Principal point ({cx}, {cy}) outside image bounds ({w}, {h})"
            )

        if rms > self.config.max_rms_error:
            raise CalibrationError(
                f"Reprojection RMS error {rms:.3f} px exceeds threshold {self.config.max_rms_error:.3f} px"
            )
