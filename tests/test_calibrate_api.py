import json
import os
import sys
from pathlib import Path
import numpy as np
import pytest

_app_dir = Path(__file__).resolve().parent.parent / "apps" / "truck_loading_detection"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import app as app_module
from app import app, state
from apps.truck_loading_detection.calibration.homography import (
    REQUIRED_LANDMARK_NAMES,
)
from apps.truck_loading_detection.calibration.storage import CalibrationStorage

SYNTHETIC_WORLD_COORDS = {
    "gate_top_left": (-1.2, 2.4),
    "gate_top_right": (1.2, 2.4),
    "gate_bottom_right": (1.2, 0.0),
    "gate_bottom_left": (-1.2, 0.0),
    "gate_mid_left": (-1.2, 1.2),
    "gate_mid_right": (1.2, 1.2),
}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_calibrate_html_dashboard_route(client):
    """Verify /calibrate/{site_id}/{dock_id} renders the dashboard."""
    resp = client.get("/calibrate/wh_lahore_01/dock_01")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Loading-Gate Calibration" in html
    assert "wh_lahore_01" in html
    assert "dock_01" in html


def test_calibrate_snapshot_endpoint_uses_requested_camera_source(client, monkeypatch):
    """Verify /api/v2/calibrate/{site_id}/{dock_id}/snapshot returns JPEG."""
    monkeypatch.setitem(state.live_cameras, "cam_top", "0")

    class FakeCapture:
        def __init__(self, source):
            assert source == 0

        def isOpened(self):
            return True

        def read(self):
            return True, np.full((720, 1280, 3), 80, dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(app_module.cv2, "VideoCapture", FakeCapture)
    resp = client.get("/api/v2/calibrate/wh_lahore_01/dock_01/snapshot?camera_id=cam_top")
    assert resp.status_code == 200
    assert resp.content_type == "image/jpeg"
    assert len(resp.data) > 1000  # Non-empty JPEG bytes


def test_calibrate_snapshot_refuses_unconfigured_camera(client):
    response = client.get("/api/v2/calibrate/wh_lahore_01/dock_01/snapshot?camera_id=cam_flank")

    assert response.status_code == 409
    assert "refusing to return another camera or a sample frame" in response.get_json()["error"]


def test_calibrate_snapshot_reuses_the_selected_live_camera_frame(client, monkeypatch):
    monkeypatch.setitem(state.live_cameras, "cam_side", "rtsp://side/stream")
    monkeypatch.setattr(
        app_module.live_camera_manager,
        "status",
        lambda: {"active": True},
    )
    monkeypatch.setattr(
        app_module.live_camera_manager,
        "latest_jpeg",
        lambda camera_id: b"\xff\xd8side-frame",
    )

    response = client.get(
        "/api/v2/calibrate/wh_lahore_01/dock_01/snapshot?camera_id=cam_side"
    )

    assert response.status_code == 200
    assert response.content_type == "image/jpeg"
    assert response.data == b"\xff\xd8side-frame"


def test_calibrate_landmarks_success_and_status(client, monkeypatch, tmp_path):
    """
    Test submitting 6 valid landmarks via API:
    - Homography estimated
    - RMS <= 2.0 px verified
    - Calibration persisted
    - Status reflects CALIBRATED_OPERATIONAL with spatial penalties active
    """
    # Create realistic landmark coordinates mapped inside 1280x720
    # X in [-1.2, 1.2] -> u in [280, 1000], Y in [0, 2.4] -> v in [560, 128]
    landmarks = {
        name: [640 + 300 * world_xy[0], 560 - 180 * world_xy[1]]
        for name, world_xy in SYNTHETIC_WORLD_COORDS.items()
    }

    validation_world = [(-0.9, 1.8), (0.9, 1.8), (-0.9, 0.2), (0.9, 0.2)]
    payload = {
        "camera_id": "cam_top",
        "image_width": 1280,
        "image_height": 720,
        "landmarks": landmarks,
        "calibration_plane": "loading_gate_plane",
        "image_orientation_confirmed": True,
        "frame_source": "local_image",
        "source_frame_name": "cam_top-frame.jpg",
        "physical_reference_points": {
            name: list(coords) for name, coords in SYNTHETIC_WORLD_COORDS.items()
        },
        "independent_validation_points": [
            {
                "world": list(world_xy),
                "image": [640 + 300 * world_xy[0], 560 - 180 * world_xy[1]],
            }
            for world_xy in validation_world
        ],
    }
    monkeypatch.setattr(app_module, "calib_storage", CalibrationStorage(tmp_path))

    resp = client.post(
        "/api/v2/calibrate/wh_test_01/dock_test_01/landmarks",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["rms_error_px"] <= 2.0, f"Expected RMS <= 2.0 px, got {data['rms_error_px']}"
    assert data["inlier_count"] >= 4

    # Verify status endpoint reflects calibrated state and active spatial penalties
    status_resp = client.get("/api/v2/calibrate/wh_test_01/dock_test_01/status")
    assert status_resp.status_code == 200
    status_data = status_resp.get_json()
    assert status_data["is_calibrated"] is False
    assert "cam_top" in status_data["cameras"]
    cam_status = status_data["cameras"]["cam_top"]
    assert cam_status["alignment_status"] == "CALIBRATED_OPERATIONAL"
    assert cam_status["spatial_penalties_enabled"] is True
    assert cam_status["independent_validation_rms_px"] <= 2.0
    assert status_data["cameras"]["cam_side"]["is_calibrated"] is False
    assert status_data["cameras"]["cam_flank"]["is_calibrated"] is False


def test_calibrate_landmarks_rejects_missing_physical_measurements(client):
    response = client.post(
        "/api/v2/calibrate/wh_test_01/dock_test_01/landmarks",
        json={
            "camera_id": "cam_side",
            "image_width": 1280,
            "image_height": 720,
            "image_orientation_confirmed": True,
            "frame_source": "local_image",
            "landmarks": {
                name: [float(i * 100 + 100), float(i * 50 + 100)]
                for i, name in enumerate(REQUIRED_LANDMARK_NAMES)
            },
        },
    )

    assert response.status_code == 400
    assert "surveyed world coordinates" in response.get_json()["reason"]


def test_calibrate_landmarks_rejection_on_invalid_geometry(client):
    """Verify that degenerate landmarks return HTTP 422 rejected."""
    # Collinear points
    degenerate = {name: [100.0 * i + 100.0, 360.0] for i, name in enumerate(REQUIRED_LANDMARK_NAMES)}
    payload = {
        "camera_id": "cam_top",
        "image_width": 1280,
        "image_height": 720,
        "landmarks": degenerate,
        "calibration_plane": "loading_gate_plane",
        "image_orientation_confirmed": True,
        "frame_source": "local_image",
        "physical_reference_points": {
            name: list(coords) for name, coords in SYNTHETIC_WORLD_COORDS.items()
        },
        "independent_validation_points": [
            {
                "world": list(world_xy),
                "image": [640 + 300 * world_xy[0], 560 - 180 * world_xy[1]],
            }
            for world_xy in [(-0.9, 1.8), (0.9, 1.8), (-0.9, 0.2), (0.9, 0.2)]
        ],
    }

    resp = client.post(
        "/api/v2/calibrate/wh_test_01/dock_test_01/landmarks",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 422
    data = resp.get_json()
    assert data["status"] == "rejected"
    assert "reason" in data


def test_calibrate_landmarks_requires_confirmed_processing_orientation(client):
    payload = {
        "camera_id": "cam_side",
        "image_width": 1280,
        "image_height": 720,
        "landmarks": {},
        "calibration_plane": "loading_gate_plane",
        "frame_source": "local_image",
        "physical_reference_points": {},
        "independent_validation_points": [],
    }

    response = client.post(
        "/api/v2/calibrate/wh_test_01/dock_test_01/landmarks",
        json=payload,
    )

    assert response.status_code == 400
    assert "same orientation" in response.get_json()["reason"]


def test_calibration_dashboard_exposes_collection_workflow_and_guide(client):
    response = client.get("/calibrate/wh_test_01/dock_test_01")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Capture or upload a frame" in html
    assert "validation points must be separate" in html.lower()
    assert "gate_mid_left" in html
    assert "warehouse measurement guide" in html.lower()
