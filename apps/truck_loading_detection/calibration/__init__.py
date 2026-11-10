"""
Calibration, Undistortion, and Watchdog Package (Phase 0).

Provides:
- Camera model configuration schema & validation
- ChArUco 9x6 board calibration
- Atomic calibration JSON persistence
- CV_16SC2 / uint16 undistortion LUT precomputation
- Frame & point geometrically consistent undistortion
- 6-landmark doorway plane homography & RMS validation (<= 2.0 px)
- 300s baseline vibration recorder & NCC watchdog monitor
- CAMERA_MISALIGNED detection with spatial penalty suppression
"""

from .camera_model import CameraModel, CameraModelRegistry
from .charuco_calibrator import CharucoCalibrator, CharucoConfig
from .storage import CalibrationStorage, DockCalibration, CameraCalibrationData
from .undistort import UndistortionLUTManager, GeometricUndistorter
from .homography import DoorwayHomographyEstimator, LandmarkSet
from .watchdog import VibrationBaselineRecorder, NCCWatchdog, CameraAlignmentState

__all__ = [
    "CameraModel",
    "CameraModelRegistry",
    "CharucoCalibrator",
    "CharucoConfig",
    "CalibrationStorage",
    "DockCalibration",
    "CameraCalibrationData",
    "UndistortionLUTManager",
    "GeometricUndistorter",
    "DoorwayHomographyEstimator",
    "LandmarkSet",
    "VibrationBaselineRecorder",
    "NCCWatchdog",
    "CameraAlignmentState",
]
