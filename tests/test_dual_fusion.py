"""Unit & Integration tests for Dual-Camera Layer-Wise Carton Counting Fusion."""

import os
import sys
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure apps/carton_counter is on sys.path
carton_app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "carton_counter"))
if carton_app_dir not in sys.path:
    sys.path.insert(0, carton_app_dir)

from detector import Detection, DetectionResult
from dual_fusion_engine import DualFusionEngine, LayerInfo, DualFusionResult
from main import app


def _make_detection(x1, y1, x2, y2, conf=0.9, class_name="carton"):
    return Detection(bbox=[x1, y1, x2, y2], confidence=conf, class_id=0, class_name=class_name)


def test_layer_clustering_standard():
    """Test clustering bounding boxes into 4 distinct horizontal layers."""
    frame_h = 800
    # Layer 1: y around 100-180 (3 cartons along width)
    l1 = [
        _make_detection(50, 100, 150, 180),
        _make_detection(160, 105, 260, 185),
        _make_detection(270, 98, 370, 178),
    ]
    # Layer 2: y around 250-330 (3 cartons along width)
    l2 = [
        _make_detection(50, 250, 150, 330),
        _make_detection(160, 255, 260, 335),
        _make_detection(270, 248, 370, 328),
    ]
    # Layer 3: y around 400-480 (3 cartons along width)
    l3 = [
        _make_detection(50, 400, 150, 480),
        _make_detection(160, 405, 260, 485),
        _make_detection(270, 398, 370, 478),
    ]
    # Layer 4: y around 550-630 (3 cartons along width)
    l4 = [
        _make_detection(50, 550, 150, 630),
        _make_detection(160, 555, 260, 635),
        _make_detection(270, 548, 370, 628),
    ]

    all_dets = l3 + l1 + l4 + l2  # Unsorted
    clusters = DualFusionEngine.cluster_layers_from_detections(all_dets, frame_h)

    assert len(clusters) == 4
    # Each layer must have 3 cartons
    for cluster in clusters:
        assert len(cluster) == 3

    # Must be ordered top-to-bottom
    avg_ys = [np.mean([(d.bbox[1] + d.bbox[3]) / 2 for d in c]) for c in clusters]
    assert avg_ys == sorted(avg_ys)


def test_align_and_multiply_equal_layers():
    """Test 4 layers: Front (3 cartons) x Side (4 cartons) = 48 total."""
    # Front: 4 layers with 3 cartons each
    front_clusters = [
        [_make_detection(0, 100, 50, 180)] * 3,
        [_make_detection(0, 250, 50, 330)] * 3,
        [_make_detection(0, 400, 50, 480)] * 3,
        [_make_detection(0, 550, 50, 630)] * 3,
    ]
    # Side: 4 layers with 4 cartons each
    side_clusters = [
        [_make_detection(0, 100, 50, 180)] * 4,
        [_make_detection(0, 250, 50, 330)] * 4,
        [_make_detection(0, 400, 50, 480)] * 4,
        [_make_detection(0, 550, 50, 630)] * 4,
    ]

    layers_info, total_count = DualFusionEngine.align_and_multiply_layers(
        front_clusters, side_clusters, 800, 800
    )

    assert len(layers_info) == 4
    assert total_count == 48  # (3 * 4) * 4 layers = 48

    for idx, l in enumerate(layers_info):
        assert l.layer_index == idx + 1
        assert l.front_count == 3
        assert l.side_count == 4
        assert l.layer_total == 12


def test_align_and_multiply_unequal_layers_fallback():
    """Test when one view has a missed/cutoff layer (e.g. 4 layers on Front, 3 on Side)."""
    # Front: 4 layers with 3 cartons each
    front_clusters = [
        [_make_detection(0, 100, 50, 180)] * 3,
        [_make_detection(0, 250, 50, 330)] * 3,
        [_make_detection(0, 400, 50, 480)] * 3,
        [_make_detection(0, 550, 50, 630)] * 3,
    ]
    # Side: 3 layers with 4 cartons each (4th layer cut off)
    side_clusters = [
        [_make_detection(0, 100, 50, 180)] * 4,
        [_make_detection(0, 250, 50, 330)] * 4,
        [_make_detection(0, 400, 50, 480)] * 4,
    ]

    layers_info, total_count = DualFusionEngine.align_and_multiply_layers(
        front_clusters, side_clusters, 800, 800
    )

    assert len(layers_info) == 4
    # The 4th layer uses median side count (4) -> 3 * 4 = 12
    assert total_count == 48


from unittest.mock import patch

def test_post_count_dual_api():
    """Test POST /count/dual endpoint with mocked detections."""
    import cv2
    client = TestClient(app)

    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)

    _, buf1 = cv2.imencode(".jpg", img1)
    _, buf2 = cv2.imencode(".jpg", img2)

    files = {
        "front": ("front.jpg", buf1.tobytes(), "image/jpeg"),
        "side": ("side.jpg", buf2.tobytes(), "image/jpeg"),
    }

    mock_front_res = DetectionResult(
        detections=[
            _make_detection(50, 100, 150, 180),
            _make_detection(160, 100, 260, 180),
            _make_detection(50, 250, 150, 330),
            _make_detection(160, 250, 260, 330),
        ],
        inference_time_ms=10.0,
        image_size=(640, 480),
        model_name="mock_model",
    )
    mock_side_res = DetectionResult(
        detections=[
            _make_detection(50, 100, 150, 180),
            _make_detection(160, 100, 260, 180),
            _make_detection(270, 100, 370, 180),
            _make_detection(50, 250, 150, 330),
            _make_detection(160, 250, 260, 330),
            _make_detection(270, 250, 370, 330),
        ],
        inference_time_ms=10.0,
        image_size=(640, 480),
        model_name="mock_model",
    )

    with patch("detector.CartonDetector.detect", side_effect=[mock_front_res, mock_side_res]):
        response = client.post("/count/dual", files=files)


    assert response.status_code == 200
    data = response.json()

    assert data["total_count"] == 12  # (2 * 3) + (2 * 3) = 12
    assert data["layers_count"] == 2
    assert len(data["layers"]) == 2
    assert data["layers"][0]["front_count"] == 2
    assert data["layers"][0]["side_count"] == 3
    assert data["layers"][0]["layer_total"] == 6
    assert data["method"] == "dual_layer_multiplication"
    assert "front_annotated_base64" in data
    assert "side_annotated_base64" in data

