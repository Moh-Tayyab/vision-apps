"""Unit tests for CameraModel schema validation and CameraModelRegistry."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from apps.truck_loading_detection.calibration.camera_model import (
    CameraModel,
    CameraModelRegistry,
    CameraModelValidationError,
)


def test_valid_pinhole_camera_model():
    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
    model = CameraModel(
        model_id="test_cam",
        camera_model="pinhole",
        image_width=1280,
        image_height=720,
        focal_length_mm=2.8,
        K=K,
        D=D,
    )
    assert model.model_id == "test_cam"
    assert model.image_width == 1280
    assert model.image_height == 720
    d = model.to_dict()
    assert d["camera_model"] == "pinhole"
    assert len(d["D"]) == 5


def test_valid_fisheye_camera_model():
    K = np.array([[500.0, 0.0, 640.0], [0.0, 500.0, 360.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.04, 0.01, -0.001, 0.0002])
    model = CameraModel(
        model_id="fish_cam",
        camera_model="fisheye",
        image_width=1280,
        image_height=720,
        focal_length_mm=1.8,
        K=K,
        D=D,
    )
    assert model.camera_model == "fisheye"
    assert len(model.D) == 4


def test_reject_nan_in_k():
    K = np.array([[np.nan, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
    with pytest.raises(CameraModelValidationError, match="non-finite"):
        CameraModel(
            model_id="bad_k",
            camera_model="pinhole",
            image_width=1280,
            image_height=720,
            focal_length_mm=2.8,
            K=K,
            D=D,
        )


def test_reject_inf_in_d():
    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.1, np.inf, 0.001, -0.001, 0.0])
    with pytest.raises(CameraModelValidationError, match="non-finite"):
        CameraModel(
            model_id="bad_d",
            camera_model="pinhole",
            image_width=1280,
            image_height=720,
            focal_length_mm=2.8,
            K=K,
            D=D,
        )


def test_reject_invalid_k_dimensions():
    K = np.array([[800.0, 0.0], [0.0, 800.0]])
    D = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
    with pytest.raises(CameraModelValidationError, match="3x3"):
        CameraModel(
            model_id="bad_shape",
            camera_model="pinhole",
            image_width=1280,
            image_height=720,
            focal_length_mm=2.8,
            K=K,
            D=D,
        )


def test_reject_negative_focal_length():
    K = np.array([[-800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
    with pytest.raises(CameraModelValidationError, match="positive"):
        CameraModel(
            model_id="neg_focal",
            camera_model="pinhole",
            image_width=1280,
            image_height=720,
            focal_length_mm=2.8,
            K=K,
            D=D,
        )


def test_reject_principal_point_out_of_bounds():
    K = np.array([[800.0, 0.0, 1500.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
    with pytest.raises(CameraModelValidationError, match="Principal point cx"):
        CameraModel(
            model_id="out_bounds",
            camera_model="pinhole",
            image_width=1280,
            image_height=720,
            focal_length_mm=2.8,
            K=K,
            D=D,
        )


def test_reject_wrong_distortion_coefficient_count():
    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    # 3 coefficients is invalid for pinhole
    D = np.array([-0.1, 0.01, 0.001])
    with pytest.raises(CameraModelValidationError, match="coefficients"):
        CameraModel(
            model_id="bad_count",
            camera_model="pinhole",
            image_width=1280,
            image_height=720,
            focal_length_mm=2.8,
            K=K,
            D=D,
        )

    # 5 coefficients is invalid for fisheye (requires 4)
    D_fish = np.array([-0.1, 0.01, 0.001, 0.002, 0.001])
    with pytest.raises(CameraModelValidationError, match="exactly 4"):
        CameraModel(
            model_id="bad_fish_count",
            camera_model="fisheye",
            image_width=1280,
            image_height=720,
            focal_length_mm=1.8,
            K=K,
            D=D_fish,
        )


def test_load_default_camera_models_yaml():
    yaml_path = Path("config/camera_models.yaml")
    assert yaml_path.exists()
    registry = CameraModelRegistry(yaml_path)
    assert "hikvision_ds2cd2143g2_28mm" in registry.models
    assert "hikvision_ds2cd2143g2_40mm" in registry.models
    assert "fisheye_wide_180" in registry.models

    m28 = registry.get("hikvision_ds2cd2143g2_28mm")
    assert m28.camera_model == "pinhole"
    assert m28.focal_length_mm == 2.8
    assert m28.K.shape == (3, 3)
    assert m28.D.size == 5

    m_fish = registry.get("fisheye_wide_180")
    assert m_fish.camera_model == "fisheye"
    assert m_fish.D.size == 4
