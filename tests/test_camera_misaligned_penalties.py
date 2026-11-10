"""
Unit tests for CAMERA_MISALIGNED detection and spatial-penalty suppression (Phase 0.10).
"""

import numpy as np
import pytest

from apps.truck_loading_detection.calibration.watchdog import (
    BaselineStatistics,
    CameraAlignmentState,
    NCCWatchdog,
)


def _make_dummy_template():
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 255
    return img


def test_spatial_penalties_disabled_on_misalignment():
    """Verify spatial penalties are enabled in operational state and disabled on misalignment."""
    baseline = BaselineStatistics(
        mu_NCC=0.90,
        sigma_NCC=0.02,
        threshold_tau=0.80,
        sample_count=40,
        duration_sec=300.0,
    )
    template = _make_dummy_template()

    watchdog = NCCWatchdog(
        camera_id="cam_left",
        site_id="wh_01",
        dock_id="dock_01",
        baseline=baseline,
        reference_template=template,
    )

    # 1. Calibrated and operational -> spatial penalties MUST BE active
    assert watchdog.state == CameraAlignmentState.CALIBRATED_OPERATIONAL
    assert watchdog.spatial_penalties_enabled is True

    # 2. Camera misaligned -> spatial penalties MUST BE disabled
    watchdog.state = CameraAlignmentState.CAMERA_MISALIGNED
    assert watchdog.spatial_penalties_enabled is False

    # 3. Transition to recalibrating -> spatial penalties MUST REMAIN disabled
    watchdog.set_recalibrating()
    assert watchdog.state == CameraAlignmentState.RECALIBRATING
    assert watchdog.spatial_penalties_enabled is False

    # 4. Failed recalibration -> remains not operational, penalties MUST REMAIN disabled
    watchdog.state = CameraAlignmentState.CALIBRATION_INVALID
    assert watchdog.spatial_penalties_enabled is False

    # 5. Successful recalibration with new validated baseline -> restored!
    new_baseline = BaselineStatistics(
        mu_NCC=0.92,
        sigma_NCC=0.015,
        threshold_tau=0.80,
        sample_count=50,
        duration_sec=300.0,
    )
    new_template = _make_dummy_template()
    watchdog.update_baseline(new_baseline, new_template)

    assert watchdog.state == CameraAlignmentState.CALIBRATED_OPERATIONAL
    assert watchdog.spatial_penalties_enabled is True


def test_spatial_penalty_suppression_in_scoring():
    """
    Demonstrate that when spatial_penalties_enabled is False, a spatial
    penalty term (e.g. corridor distance penalty) is neutralized (set to 0.0),
    preventing valid detections from being dropped due to camera misalignment.
    """
    class ScoringPipeline:
        def __init__(self, watchdog: NCCWatchdog):
            self.watchdog = watchdog

        def compute_detection_score(self, raw_confidence: float, corridor_dist_px: float) -> float:
            base_score = raw_confidence
            if self.watchdog.spatial_penalties_enabled:
                # Apply 0.05 penalty per 10 pixels off corridor axis
                spatial_penalty = (corridor_dist_px / 10.0) * 0.05
            else:
                # SUPPRESSED when camera is misaligned
                spatial_penalty = 0.0

            return max(0.0, base_score - spatial_penalty)

    baseline = BaselineStatistics(0.90, 0.02, 0.80, 50, 300.0)
    watchdog = NCCWatchdog("cam_01", "site", "dock", baseline, _make_dummy_template())
    pipeline = ScoringPipeline(watchdog)

    # When calibrated: 50px offset penalizes score by 0.25
    score_calibrated = pipeline.compute_detection_score(raw_confidence=0.85, corridor_dist_px=50.0)
    assert pytest.approx(score_calibrated) == 0.60

    # When camera is misaligned: spatial penalty is SUPPRESSED
    watchdog.state = CameraAlignmentState.CAMERA_MISALIGNED
    score_misaligned = pipeline.compute_detection_score(raw_confidence=0.85, corridor_dist_px=50.0)
    assert pytest.approx(score_misaligned) == 0.85  # Penalty neutralized!
