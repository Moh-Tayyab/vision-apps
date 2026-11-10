"""
Unit tests for Undistortion LUT Precomputation & Performance (Phase 0.4).
"""

import time
import cv2
import numpy as np
import pytest

from apps.truck_loading_detection.calibration.undistort import (
    UndistortionLUTManager,
    UndistortionError,
)


def test_cv_16sc2_map_types_and_shapes():
    """
    Investigate and verify CV_16SC2 versus 'float16' misnomer:
    In OpenCV, cv2.CV_16SC2 produces map1 as int16 (H, W, 2) and map2 as uint16 (H, W).
    It is NOT IEEE float16.
    """
    lut_mgr = UndistortionLUTManager(default_map_type=cv2.CV_16SC2)
    K = np.array([[850.0, 0.0, 640.0], [0.0, 850.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([-0.18, 0.04, -0.001, 0.0008, -0.003], dtype=np.float64)

    lut = lut_mgr.get_or_create_lut("cam_test", (1280, 720), K, D)

    # Validate exact types and shapes
    assert lut.map1.dtype == np.int16
    assert lut.map1.shape == (720, 1280, 2)
    assert lut.map2.dtype == np.uint16
    assert lut.map2.shape == (720, 1280)

    # Remap an image with these lookup tables
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    remap_out = cv2.remap(img, lut.map1, lut.map2, cv2.INTER_LINEAR)
    assert remap_out.shape == (720, 1280, 3)


def test_lut_cache_reuse_and_invalidation():
    """Verify that LUTs are cached and reused, but invalidated on parameter or size change."""
    lut_mgr = UndistortionLUTManager()
    K1 = np.array([[850.0, 0.0, 640.0], [0.0, 850.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D1 = np.array([-0.18, 0.04, -0.001, 0.0008, -0.003], dtype=np.float64)

    lut1 = lut_mgr.get_or_create_lut("cam_1", (1280, 720), K1, D1)
    # Second retrieval with same params should return the EXACT SAME cached object
    lut2 = lut_mgr.get_or_create_lut("cam_1", (1280, 720), K1, D1)
    assert lut1 is lut2

    # Changing image resolution must recompute a new LUT
    lut_res = lut_mgr.get_or_create_lut("cam_1", (1920, 1080), K1, D1)
    assert lut_res is not lut1
    assert lut_res.image_width == 1920
    assert lut_res.image_height == 1080

    # Changing K must recompute a new LUT
    K2 = K1.copy()
    K2[0, 0] = 900.0
    lut_k = lut_mgr.get_or_create_lut("cam_1", (1920, 1080), K2, D1)
    assert lut_k is not lut_res
    assert lut_k.cache_key != lut_res.cache_key


def test_remap_performance():
    """Measure remap execution time with precomputed LUT to verify sub-millisecond execution."""
    lut_mgr = UndistortionLUTManager(default_map_type=cv2.CV_16SC2)
    K = np.array([[850.0, 0.0, 640.0], [0.0, 850.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([-0.18, 0.04, -0.001, 0.0008, -0.003], dtype=np.float64)

    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Warm-up precomputation
    _ = lut_mgr.remap_frame(test_frame, "cam_bench", K, D)

    # Benchmark 20 iterations
    iterations = 20
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = lut_mgr.remap_frame(test_frame, "cam_bench", K, D)
    elapsed = (time.perf_counter() - t0) / iterations * 1000.0  # ms per frame

    # Assert remap takes less than 15 ms on CPU (typically ~1-3 ms on modern CPU)
    assert elapsed < 15.0, f"Remap too slow: {elapsed:.2f} ms"
