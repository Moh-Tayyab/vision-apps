"""
Unit tests for ChArUco 9x6 calibration pipeline (Phase 0.2).
"""

import cv2
import numpy as np
import pytest

from apps.truck_loading_detection.calibration.charuco_calibrator import (
    CalibrationError,
    CharucoCalibrator,
    CharucoConfig,
)


def test_charuco_board_geometry():
    """Verify standard 9x6 board geometry with 40 interior corners."""
    config = CharucoConfig(squares_x=9, squares_y=6, square_length_m=0.04, marker_length_m=0.02)
    board = config.create_board()
    assert board is not None
    corners = board.getChessboardCorners()
    # 9x6 chessboard squares -> (9 - 1) * (6 - 1) = 8 * 5 = 40 interior corners
    assert corners.shape == (40, 3)


def test_charuco_corner_detection_on_board_image():
    """Test corner detection on an ideal rendered board."""
    config = CharucoConfig(squares_x=9, squares_y=6, square_length_m=0.04, marker_length_m=0.02)
    calibrator = CharucoCalibrator(config)
    board_img = calibrator.board.generateImage((1280, 853))

    corners, ids = calibrator.detect_corners(board_img)
    assert corners is not None
    assert ids is not None
    assert len(corners) == 40
    assert len(ids) == 40


def test_synthetic_calibration_pipeline():
    """
    Test calibration recovery using synthetic 3D->2D point correspondences
    with known ground truth camera intrinsics K and distortion D.
    """
    config = CharucoConfig(squares_x=9, squares_y=6, min_views=5, max_rms_error=1.5)
    calibrator = CharucoCalibrator(config)

    # Known ground truth camera
    true_K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    true_D = np.array([-0.05, 0.01, 0.001, -0.001, 0.0], dtype=np.float64)
    img_size = (1280, 720)

    board_corners_3d = calibrator.board.getChessboardCorners()

    # Generate 6 synthetic views with varying poses
    np.random.seed(42)
    for i in range(6):
        # Rotation vector (small rotations around X, Y, Z)
        rvec = np.array([0.1 * (i - 2.5), 0.15 * np.cos(i), 0.05 * np.sin(i)], dtype=np.float64)
        # Translation vector (Z distance ~ 0.8 to 1.2m)
        tvec = np.array([0.05 * (i - 2.5), -0.02 * i, 0.85 + 0.05 * i], dtype=np.float64)

        # Project 3D points
        proj_pts, _ = cv2.projectPoints(board_corners_3d, rvec, tvec, true_K, true_D)
        proj_pts = proj_pts.reshape(-1, 2)

        # Add small subpixel noise (0.05 px)
        noise = np.random.normal(0, 0.05, proj_pts.shape)
        noisy_pts = (proj_pts + noise).astype(np.float32)

        calibrator.add_corners(board_corners_3d, noisy_pts, img_size)

    # Run calibration
    result = calibrator.calibrate("pinhole")
    assert result.num_views_used == 6
    assert result.rms_error < 0.5  # Sub-pixel accuracy on clean synthetic data
    assert np.allclose(result.K[0, 0], true_K[0, 0], rtol=0.05)
    assert np.allclose(result.K[1, 1], true_K[1, 1], rtol=0.05)
    assert np.allclose(result.K[0, 2], true_K[0, 2], rtol=0.05)
    assert np.allclose(result.K[1, 2], true_K[1, 2], rtol=0.05)
    assert len(result.per_view_errors) == 6


def test_calibration_fails_with_insufficient_views():
    """Verify that calibrate() raises CalibrationError when fewer than min_views are provided."""
    config = CharucoConfig(min_views=5)
    calibrator = CharucoCalibrator(config)

    board_corners_3d = calibrator.board.getChessboardCorners()
    # Add only 3 views
    for _ in range(3):
        calibrator.add_corners(board_corners_3d, board_corners_3d[:, :2], (1280, 720))

    with pytest.raises(CalibrationError, match="Insufficient usable calibration views"):
        calibrator.calibrate("pinhole")
