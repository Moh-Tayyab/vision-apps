"""
Unit tests for Vibration Baseline and NCC Watchdog (Phase 0.8 & 0.9).
"""

import cv2
import numpy as np
import pytest

from apps.truck_loading_detection.calibration.watchdog import (
    BaselineStatistics,
    CameraAlignmentState,
    NCCWatchdog,
    VibrationBaselineRecorder,
    compute_edge_ncc,
)


class MockClock:
    """Controllable monotonic clock for deterministic time testing."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.current_time = start_time

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


def test_edge_ncc_identical_and_shifted():
    """Verify compute_edge_ncc returns 1.0 for identical edges and lower for shifted."""
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), 255, 3)
    edges = cv2.Canny(img, 50, 150)

    # Identical
    ncc_self = compute_edge_ncc(edges, edges)
    assert pytest.approx(ncc_self, abs=1e-3) == 1.0

    # Shifted image
    shifted_img = np.roll(img, shift=30, axis=1)
    shifted_edges = cv2.Canny(shifted_img, 50, 150)
    ncc_shifted = compute_edge_ncc(edges, shifted_edges)
    assert ncc_shifted < 0.5


def test_vibration_baseline_300s_recording():
    """Test full 300-second baseline recording using injected controllable clock."""
    clock = MockClock(1000.0)
    recorder = VibrationBaselineRecorder(
        camera_id="cam_top",
        site_id="wh_01",
        dock_id="dock_01",
        target_duration_sec=300.0,
        min_samples=30,
        clock_fn=clock,
    )

    base_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(base_frame, (300, 100), (980, 620), (255, 255, 255), 4)

    recorder.start(base_frame)

    # Simulate 50 frames over 300 seconds with slight vibration jitter
    np.random.seed(42)
    for i in range(50):
        clock.advance(6.0)  # 50 * 6s = 300s
        jitter_x = int(np.random.choice([-1, 0, 1]))
        frame = np.roll(base_frame, shift=jitter_x, axis=1)
        done = recorder.record_frame(frame)
        if i == 49:
            assert done is True

    stats = recorder.finalize()
    assert stats.sample_count >= 50
    assert stats.duration_sec >= 300.0
    assert 0.80 <= stats.mu_NCC <= 1.0
    assert stats.sigma_NCC >= 0.0

    # Verify formula: tau = min(0.80, mu - 4 * sigma)
    expected_tau = min(0.80, stats.mu_NCC - 4.0 * stats.sigma_NCC)
    assert pytest.approx(stats.threshold_tau, abs=1e-4) == expected_tau
    assert stats.threshold_tau <= 0.80  # Upper cap at 0.80


def test_watchdog_triggers_camera_misaligned():
    """Verify watchdog detects camera shift and emits CAMERA_MISALIGNED alert."""
    clock = MockClock(2000.0)
    base_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(base_frame, (300, 100), (980, 620), (255, 255, 255), 4)
    from apps.truck_loading_detection.calibration.watchdog import extract_structural_edges
    ref_template = extract_structural_edges(cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY))

    # High baseline: mu=0.92, sigma=0.02 -> tau = min(0.80, 0.92 - 0.08) = 0.80
    baseline = BaselineStatistics(
        mu_NCC=0.92,
        sigma_NCC=0.02,
        threshold_tau=0.80,
        sample_count=50,
        duration_sec=300.0,
    )

    watchdog = NCCWatchdog(
        camera_id="cam_top",
        site_id="wh_01",
        dock_id="dock_01",
        baseline=baseline,
        reference_template=ref_template,
        rolling_window_sec=60.0,
        clock_fn=clock,
    )
    assert watchdog.state == CameraAlignmentState.CALIBRATED_OPERATIONAL
    assert watchdog.spatial_penalties_enabled is True

    # 1. Normal frames: similarity ~ 1.0 > 0.80
    for _ in range(5):
        clock.advance(1.0)
        state, alert = watchdog.check_frame(base_frame)
        assert state == CameraAlignmentState.CALIBRATED_OPERATIONAL
        assert alert is None
        assert watchdog.spatial_penalties_enabled is True

    # 2. Camera gets bumped! Shift frame by 60 pixels
    bumped_frame = np.roll(base_frame, shift=60, axis=1)

    # Feed multiple shifted frames over 10 seconds
    for _ in range(10):
        clock.advance(1.0)
        state, alert = watchdog.check_frame(bumped_frame)

    assert state == CameraAlignmentState.CAMERA_MISALIGNED
    assert watchdog.spatial_penalties_enabled is False
    assert alert is not None
    assert alert.alert_type == "CAMERA_MISALIGNED"
    assert alert.camera_id == "cam_top"
    assert alert.current_similarity < 0.80
    assert alert.spatial_penalties_disabled is True
