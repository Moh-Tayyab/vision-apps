"""
Unit & Integration tests for Unified Inference Engine Manager & Fallback (Phase 1.7, 1.8, 1.12).
"""

import pytest
import numpy as np
import os
from apps.truck_loading_detection.model_pipeline.engine_manager import (
    InferenceEngineManager,
    EngineInitializationError,
    EngineBenchmarkResult,
)
from apps.truck_loading_detection.model_pipeline.class_contract import (
    CLASS_CARTON,
    CLASS_PERSON,
    DetectionBox,
)


def test_engine_manager_cpu_fallback():
    """Verify that when TensorRT/GPU is unavailable, manager falls back gracefully to PyTorch CPU."""
    manager = InferenceEngineManager(
        preferred_profile="gpu_tensorrt",
        trt_engine_path="nonexistent_engine.engine",
        pytorch_model_path="apps/truck_loading_detection/yolo11n.pt",
    )

    assert manager.active_profile == "pytorch_cpu"
    assert manager.device == "cpu"
    assert manager.model is not None


def test_engine_manager_predict_contract():
    """Verify inference returns separate carton and person lists conforming to class contract."""
    manager = InferenceEngineManager(
        pytorch_model_path="apps/truck_loading_detection/yolo11n.pt"
    )

    # Synthetic black frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cartons, persons = manager.predict(frame)

    assert isinstance(cartons, list)
    assert isinstance(persons, list)

    for c in cartons:
        assert isinstance(c, DetectionBox)
        assert c.class_id == CLASS_CARTON
        assert 0.0 <= c.confidence <= 1.0

    for p in persons:
        assert isinstance(p, DetectionBox)
        assert p.class_id == CLASS_PERSON
        assert 0.0 <= p.confidence <= 1.0


def test_engine_manager_benchmark():
    """Verify benchmark functionality computes latency and throughput."""
    manager = InferenceEngineManager(
        pytorch_model_path="apps/truck_loading_detection/yolo11n.pt"
    )

    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    result = manager.benchmark(test_frame=frame, num_iterations=3)

    assert isinstance(result, EngineBenchmarkResult)
    assert result.num_frames == 3
    assert result.mean_latency_ms > 0.0
    assert result.throughput_fps > 0.0
    assert result.engine_profile == "pytorch_cpu"


def test_engine_manager_invalid_path():
    """Verify error raised when all candidate model weights are invalid."""
    with pytest.raises(EngineInitializationError):
        InferenceEngineManager(
            preferred_profile="pytorch_cpu",
            pytorch_model_path="completely_invalid_path.pt",
        )


def test_engine_manager_filters_unsupported_classes():
    """Regression test: Ensure non-contract classes (e.g. COCO bicycle) are not misclassified as person."""
    manager = InferenceEngineManager(
        pytorch_model_path="apps/truck_loading_detection/yolo11n.pt"
    )

    # Mock an ultralytics result object with names and boxes data
    class MockBoxes:
        def __init__(self, data_tensor):
            self.data = data_tensor

        def __len__(self):
            return len(self.data)

    class MockResult:
        def __init__(self):
            # Simulated predictions:
            # Row 0: bbox [10, 10, 50, 50], conf 0.9, class 0 (person in COCO)
            # Row 1: bbox [20, 20, 60, 60], conf 0.85, class 1 (bicycle in COCO)
            # Row 2: bbox [30, 30, 70, 70], conf 0.80, class 2 (car in COCO)
            import torch
            self.boxes = MockBoxes(torch.tensor([
                [10.0, 10.0, 50.0, 50.0, 0.90, 0.0],
                [20.0, 20.0, 60.0, 60.0, 0.85, 1.0],
                [30.0, 30.0, 70.0, 70.0, 0.80, 2.0],
            ]))
            self.names = {0: "person", 1: "bicycle", 2: "car"}

    # Mock manager.model call
    original_model = manager.model
    try:
        manager.model = lambda *args, **kwargs: [MockResult()]
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        cartons, persons = manager.predict(frame)

        # Class 0 ("person") should be mapped to person
        assert len(persons) == 1
        assert persons[0].class_id == CLASS_PERSON
        assert persons[0].confidence == pytest.approx(0.90, rel=1e-3)

        # Class 1 ("bicycle") and Class 2 ("car") must NOT be treated as person or carton
        assert len(cartons) == 0
    finally:
        manager.model = original_model

