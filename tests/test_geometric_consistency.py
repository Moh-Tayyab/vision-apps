"""
Unit tests for Frame and Point Geometric Consistency (Phase 0.5).
"""

import cv2
import numpy as np
import pytest

from apps.truck_loading_detection.calibration.undistort import GeometricUndistorter


def test_point_undistortion_matches_rectified_image_geometry():
    """
    Mathematical verification of geometric consistency:
    1. Define known 3D points.
    2. Project to distorted pixel coordinates using known K and D.
    3. Undistort points with P=K_new using GeometricUndistorter.
    4. Ideal pinhole projection using K_new (with D=0) must match undistorted points!
    """
    K = np.array([[850.0, 0.0, 640.0], [0.0, 850.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([-0.18, 0.04, -0.001, 0.0008, -0.003], dtype=np.float64)
    K_new = K.copy()

    undistorter = GeometricUndistorter(
        camera_id="cam_geom",
        image_size=(1280, 720),
        K=K,
        D=D,
        K_new=K_new,
        camera_model="pinhole",
    )

    # 3D points in camera frame (Z = 2.0 meters, varying X and Y)
    pts_3d = np.array(
        [
            [-0.5, -0.3, 2.0],
            [0.0, 0.0, 2.0],
            [0.5, 0.3, 2.0],
            [-0.8, 0.4, 2.5],
            [0.8, -0.4, 2.5],
        ],
        dtype=np.float64,
    )

    # 1. Project through distorted lens model
    distorted_proj, _ = cv2.projectPoints(
        pts_3d, np.zeros(3), np.zeros(3), K, D
    )
    distorted_pts = distorted_proj.reshape(-1, 2)

    # 2. Undistort points via GeometricUndistorter (using P=K_new)
    undistorted_pts = undistorter.undistort_points(distorted_pts)

    # 3. Compute ground truth ideal pinhole projections using K_new and D=0
    zero_d = np.zeros(5, dtype=np.float64)
    ideal_proj, _ = cv2.projectPoints(
        pts_3d, np.zeros(3), np.zeros(3), K_new, zero_d
    )
    ideal_pts = ideal_proj.reshape(-1, 2)

    # 4. Point difference must be negligible (< 0.1 px)
    diff = np.abs(undistorted_pts - ideal_pts)
    assert np.all(diff < 0.1), f"Discrepancy too large: max diff = {np.max(diff)} px"


def test_omitting_p_yields_normalized_coordinates_not_pixels():
    """
    Demonstrate why cv2.undistortPoints without P is geometrically incompatible
    with pixel-space images (it returns normalized ray coordinates).
    """
    K = np.array([[850.0, 0.0, 640.0], [0.0, 850.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([-0.18, 0.04, 0.0, 0.0, 0.0], dtype=np.float64)

    pts = np.array([[[640.0, 360.0]], [[200.0, 150.0]]], dtype=np.float32)

    # Undistort WITHOUT P -> normalized ray coordinates
    norm_pts = cv2.undistortPoints(pts, K, D)
    # Undistort WITH P=K -> pixel coordinates
    pixel_pts = cv2.undistortPoints(pts, K, D, P=K)

    # Normalized points are around 0 (e.g. [-1, 1]), while pixel points are in [0, 1280] x [0, 720]
    assert np.all(np.abs(norm_pts) < 2.0)
    assert np.all(pixel_pts > 50.0)
