import json

import numpy as np

from apps.truck_loading_detection.three_camera_system.geometric_adapter import (
    GeometricCalibrationManager,
)


def _calibration_record(homography):
    return {
        "cameras": {
            "cam_top": {
                "camera_id": "cam_top",
                "image_width": 1280,
                "image_height": 720,
                "homography": homography,
            }
        }
    }


def test_checked_in_calibration_is_not_accepted_without_physical_validation():
    manager = GeometricCalibrationManager(
        "config/calibrations/wh_test_01_dock_test_01_v2.json"
    )

    assert not manager.is_fully_calibrated(["cam_top", "cam_side", "cam_flank"])
    assert not manager.profiles["cam_top"].is_calibrated
    assert manager.project("cam_top", (640, 470), image_size=(1280, 720)) is None


def test_loading_gate_projection_requires_validated_evidence_and_matching_resolution(tmp_path):
    homography = {
        "H": np.eye(3).tolist(),
        "H_inv": np.eye(3).tolist(),
        "is_valid": True,
        "rms_error_px": 0.4,
        "calibration_plane": "loading_gate_plane",
        "image_orientation_confirmed": True,
        "physical_reference_points": {
            "gate_top_left": [0.0, 0.0],
            "gate_top_right": [1.0, 0.0],
            "gate_bottom_right": [1.0, 1.0],
            "gate_bottom_left": [0.0, 1.0],
            "gate_mid_left": [0.5, 0.2],
            "gate_mid_right": [0.4, 0.8],
        },
        "independent_validation": {
            "passed": True,
            "calibration_plane": "loading_gate_plane",
            "point_count": 4,
            "rms_error_px": 0.7,
            "max_error_px": 1.1,
        },
    }
    calibration_file = tmp_path / "calibration.json"
    calibration_file.write_text(json.dumps(_calibration_record(homography)))

    manager = GeometricCalibrationManager(str(calibration_file))

    assert manager.is_fully_calibrated(["cam_top"])
    assert manager.project("cam_top", (15, 25), image_size=(1280, 720)) == (15.0, 25.0)
    assert manager.project("cam_top", (15, 25), image_size=(720, 1280)) is None

    homography["image_orientation_confirmed"] = False
    calibration_file.write_text(json.dumps(_calibration_record(homography)))
    unconfirmed_orientation = GeometricCalibrationManager(str(calibration_file))
    assert not unconfirmed_orientation.is_fully_calibrated(["cam_top"])


def test_unvalidated_or_wrong_plane_homography_is_rejected(tmp_path):
    homography = {
        "H": np.eye(3).tolist(),
        "H_inv": np.eye(3).tolist(),
        "is_valid": True,
        "rms_error_px": 0.0,
    }
    calibration_file = tmp_path / "calibration.json"
    calibration_file.write_text(json.dumps(_calibration_record(homography)))

    manager = GeometricCalibrationManager(str(calibration_file))

    assert not manager.is_fully_calibrated(["cam_top"])
    assert manager.project("cam_top", (15, 25)) is None
