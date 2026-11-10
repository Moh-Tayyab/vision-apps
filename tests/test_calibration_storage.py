"""
Unit tests for Calibration persistence & JSON schema validation (Phase 0.3).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from apps.truck_loading_detection.calibration.storage import (
    CalibrationStorage,
    CalibrationStorageError,
    CameraCalibrationData,
    DockCalibration,
)


def _make_sample_dock():
    dock = DockCalibration(site_id="wh_lahore_01", dock_id="dock_01")
    K = np.array([[850.0, 0.0, 640.0], [0.0, 850.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([-0.18, 0.04, -0.001, 0.0008, -0.003], dtype=np.float64)
    cam_top = CameraCalibrationData(
        camera_id="cam_top",
        camera_model="pinhole",
        image_width=1280,
        image_height=720,
        focal_length_mm=2.8,
        K=K,
        D=D,
        rms_error=0.45,
        per_view_errors=[0.42, 0.48, 0.44],
    )
    dock.add_camera(cam_top)
    return dock


def test_atomic_save_and_load(tmp_path):
    storage = CalibrationStorage(base_dir=tmp_path)
    dock = _make_sample_dock()

    saved_path = storage.save(dock)
    assert saved_path.exists()
    assert saved_path.name == "wh_lahore_01_dock_01_v2.json"

    # Load and verify contents
    loaded = storage.load("wh_lahore_01", "dock_01")
    assert loaded.site_id == "wh_lahore_01"
    assert loaded.dock_id == "dock_01"
    assert loaded.schema_version == "2.0"
    assert "cam_top" in loaded.cameras

    cam = loaded.get_camera("cam_top")
    assert cam.camera_id == "cam_top"
    assert cam.image_width == 1280
    assert cam.image_height == 720
    assert cam.focal_length_mm == 2.8
    assert np.allclose(cam.K, dock.cameras["cam_top"].K)
    assert np.allclose(cam.D, dock.cameras["cam_top"].D)
    assert cam.rms_error == 0.45


def test_reject_non_finite_float_on_save(tmp_path):
    storage = CalibrationStorage(base_dir=tmp_path)
    dock = _make_sample_dock()
    dock.cameras["cam_top"].K[0, 0] = np.nan

    with pytest.raises(CalibrationStorageError, match="Non-finite float"):
        storage.save(dock)

    # Ensure no partial corrupted file is left behind
    assert not storage.get_path("wh_lahore_01", "dock_01").exists()


def test_reject_malformed_json(tmp_path):
    storage = CalibrationStorage(base_dir=tmp_path)
    bad_file = tmp_path / "wh_lahore_01_dock_01_v2.json"
    bad_file.write_text("{ incomplete json ...", encoding="utf-8")

    with pytest.raises(CalibrationStorageError, match="Malformed JSON"):
        storage.load("wh_lahore_01", "dock_01")


def test_reject_mismatched_site_dock(tmp_path):
    storage = CalibrationStorage(base_dir=tmp_path)
    dock = _make_sample_dock()
    storage.save(dock)

    # Attempting to load as dock_02 should fail
    target = tmp_path / "wh_lahore_01_dock_02_v2.json"
    target.write_text((tmp_path / "wh_lahore_01_dock_01_v2.json").read_text(), encoding="utf-8")

    with pytest.raises(CalibrationStorageError, match="site/dock mismatch"):
        storage.load("wh_lahore_01", "dock_02")
