"""
Unit tests for 6-Landmark Doorway Homography and Reprojection Validation (Phase 0.7).
"""

import numpy as np
import pytest

from apps.truck_loading_detection.calibration.homography import (
    DoorwayHomographyEstimator,
    HomographyValidationError,
    LandmarkSet,
    REQUIRED_LANDMARK_NAMES,
)

SYNTHETIC_WORLD_COORDS = {
    "gate_top_left": (-1.2, 2.4),
    "gate_top_right": (1.2, 2.4),
    "gate_bottom_right": (1.2, 0.0),
    "gate_bottom_left": (-1.2, 0.0),
    "gate_mid_left": (-1.2, 1.2),
    "gate_mid_right": (1.2, 1.2),
}


def _generate_synthetic_landmarks(noise_std: float = 0.0) -> LandmarkSet:
    """Generate synthetic gate-plane calibration observations for unit tests."""
    H_world_to_img = np.array(
        [
            [300.0, 5.0, 640.0],
            [-5.0, -180.0, 560.0],
            [0.002, -0.003, 1.0],
        ],
        dtype=np.float64,
    )

    np.random.seed(42)
    pts = {}
    for name in REQUIRED_LANDMARK_NAMES:
        wx, wy = SYNTHETIC_WORLD_COORDS[name]
        world_pt = np.array([wx, wy, 1.0], dtype=np.float64)
        img_homo = H_world_to_img @ world_pt
        u = img_homo[0] / img_homo[2]
        v = img_homo[1] / img_homo[2]
        if noise_std > 0:
            u += np.random.normal(0, noise_std)
            v += np.random.normal(0, noise_std)
        pts[name] = (float(u), float(v))

    return LandmarkSet(points=pts, image_width=1280, image_height=720)


def test_homography_valid_rms_under_2px():
    """Verify that accurate landmark annotations achieve RMS <= 2.0 px and pass validation."""
    estimator = DoorwayHomographyEstimator(SYNTHETIC_WORLD_COORDS, max_reprojection_rms=2.0)
    # Add small 0.5 px realistic click jitter
    landmarks = _generate_synthetic_landmarks(noise_std=0.5)

    result = estimator.estimate(landmarks)
    assert result.is_valid is True
    assert result.rms_error_px <= 2.0, f"Expected RMS <= 2.0, got {result.rms_error_px}"
    assert result.inlier_count == 6
    assert result.inlier_ratio == 1.0


def test_homography_rejection_when_rms_exceeds_2px():
    """Verify that excessive noise causing RMS > 2.0 px fails the acceptance gate."""
    estimator = DoorwayHomographyEstimator(SYNTHETIC_WORLD_COORDS, max_reprojection_rms=2.0)
    # Excessive 4.5 px noise
    landmarks = _generate_synthetic_landmarks(noise_std=4.5)

    result = estimator.estimate(landmarks)
    assert result.is_valid is False
    assert result.rms_error_px > 2.0
    assert "exceeds strict threshold" in str(result.rejection_reason)


def test_missing_landmark_raises_error():
    """Verify that providing fewer than 6 landmarks raises HomographyValidationError."""
    incomplete_pts = {
        "gate_top_left": (200.0, 100.0),
        "gate_top_right": (1000.0, 100.0),
        "gate_bottom_right": (1000.0, 600.0),
        # missing gate_bottom_left, gate_mid_left, gate_mid_right
    }
    with pytest.raises(HomographyValidationError, match="Missing required landmarks"):
        LandmarkSet(points=incomplete_pts, image_width=1280, image_height=720)


def test_collinear_degenerate_landmarks_rejected():
    """Verify that degenerate collinear landmarks produce invalid homography."""
    estimator = DoorwayHomographyEstimator(SYNTHETIC_WORLD_COORDS, max_reprojection_rms=2.0)
    # All points placed in a single straight line
    degenerate_pts = {name: (100.0 * i + 100.0, 360.0) for i, name in enumerate(REQUIRED_LANDMARK_NAMES)}
    landmarks = LandmarkSet(points=degenerate_pts, image_width=1280, image_height=720)

    result = estimator.estimate(landmarks)
    assert result.is_valid is False
