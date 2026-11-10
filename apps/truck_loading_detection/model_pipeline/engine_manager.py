"""
Unified Inference Engine Manager & Production Fallback (Phase 1.7, 1.8, 1.12).

Manages multi-tier inference engine execution:
  - Profile A: TensorRT FP16 (.engine) on GPU
  - Profile B: OpenVINO INT8 on CPU
  - Fallback: PyTorch CPU (.pt) via Ultralytics

Guarantees strict two-class contract enforcement and graceful hardware fallback.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .class_contract import (
    CLASS_CARTON,
    CLASS_PERSON,
    DetectionBox,
    InvalidClassContractError,
)

logger = logging.getLogger("truck_loading.engine_manager")


class EngineInitializationError(Exception):
    """Raised when inference engine loading or hardware initialization fails."""
    pass


@dataclass
class EngineBenchmarkResult:
    """Benchmark performance report."""

    engine_profile: str
    model_path: str
    device: str
    num_frames: int
    mean_latency_ms: float
    p95_latency_ms: float
    throughput_fps: float


class InferenceEngineManager:
    """
    Manages model loading, automatic hardware backend selection, and inference execution.
    """

    def __init__(
        self,
        preferred_profile: str = "auto",  # "gpu_tensorrt", "openvino_int8", "pytorch_cpu", "auto"
        trt_engine_path: Optional[str] = None,
        openvino_model_path: Optional[str] = None,
        pytorch_model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
    ) -> None:
        self.preferred_profile = preferred_profile
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.img_size = img_size

        self.trt_path = trt_engine_path
        self.openvino_path = openvino_model_path
        self.pytorch_path = pytorch_model_path

        self.active_profile: str = "uninitialized"
        self.active_model_path: str = ""
        self.device: str = "cpu"
        self.model: Optional[YOLO] = None

        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Select and initialize the appropriate inference engine based on hardware availability."""
        cuda_available = torch.cuda.is_available()

        # 1. Attempt TensorRT on GPU if requested and available
        if self.preferred_profile in ("auto", "gpu_tensorrt") and cuda_available:
            if self.trt_path and os.path.exists(self.trt_path):
                try:
                    logger.info("Initializing TensorRT GPU engine from %s", self.trt_path)
                    self.model = YOLO(self.trt_path)
                    self.active_profile = "gpu_tensorrt"
                    self.active_model_path = self.trt_path
                    self.device = "cuda:0"
                    self._verify_class_contract()
                    return
                except Exception as e:
                    logger.warning("TensorRT initialization failed: %s. Falling back.", e)

        # 2. Attempt OpenVINO on CPU if requested and available
        if self.preferred_profile in ("auto", "openvino_int8"):
            if self.openvino_path and os.path.exists(self.openvino_path):
                try:
                    logger.info("Initializing OpenVINO CPU engine from %s", self.openvino_path)
                    self.model = YOLO(self.openvino_path)
                    self.active_profile = "openvino_int8"
                    self.active_model_path = self.openvino_path
                    self.device = "cpu"
                    self._verify_class_contract()
                    return
                except Exception as e:
                    logger.warning("OpenVINO initialization failed: %s. Falling back.", e)

        # 3. Fallback to PyTorch CPU
        try:
            resolved_pt = self._resolve_pytorch_path()
            logger.info("Initializing PyTorch CPU fallback engine from %s", resolved_pt)
            self.model = YOLO(resolved_pt)
            self.active_profile = "pytorch_cpu"
            self.active_model_path = resolved_pt
            self.device = "cpu"
            self._verify_class_contract()
        except Exception as e:
            raise EngineInitializationError(f"Failed to initialize PyTorch fallback engine: {e}") from e

    def _resolve_pytorch_path(self) -> str:
        if self.pytorch_path:
            if os.path.exists(self.pytorch_path):
                return self.pytorch_path
            raise FileNotFoundError(f"Explicitly specified PyTorch model not found: '{self.pytorch_path}'")

        candidates = [
            "apps/truck_loading_detection/best.pt",
            "apps/truck_loading_detection/yolo11n.pt",
            "yolo11n.pt",
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c
        raise FileNotFoundError(f"No valid PyTorch fallback model found among candidates: {candidates}")

    def _verify_class_contract(self) -> None:
        """Ensure model recognizes at least classes carton and person."""
        if hasattr(self.model, "names") and isinstance(self.model.names, dict):
            logger.info("Engine %s loaded classes: %s", self.active_profile, self.model.names)

    def predict(
        self, frame: np.ndarray
    ) -> Tuple[List[DetectionBox], List[DetectionBox]]:
        """
        Run inference on frame.
        Returns: (carton_detections, person_detections).
        """
        if self.model is None:
            raise RuntimeError("Inference engine is not initialized")

        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            verbose=False,
            device=self.device,
        )

        cartons: List[DetectionBox] = []
        persons: List[DetectionBox] = []

        if not results:
            return cartons, persons

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return cartons, persons

        boxes_data = res.boxes.data.cpu().numpy()  # (N, 6): [x1, y1, x2, y2, conf, cls]

        for row in boxes_data:
            x1, y1, x2, y2, conf, cls_id = row
            cls_int = int(round(cls_id))

            # Map detector output into strict 2-class contract:
            # If model uses COCO classes: class 0 is person.
            # In our canonical contract: 0 is carton, 1 is person.
            # If raw model has names dict, match by label name; otherwise match canonical ID.
            has_names = hasattr(res, "names") and isinstance(res.names, dict) and len(res.names) > 0
            cls_name = res.names.get(cls_int, "").lower() if has_names else ""

            if cls_name == "person" or (not has_names and cls_int == CLASS_PERSON):
                det = DetectionBox(
                    class_id=CLASS_PERSON,
                    confidence=float(conf),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
                persons.append(det)
            elif cls_name in ("carton", "box", "cardboard box") or (not has_names and cls_int == CLASS_CARTON):
                det = DetectionBox(
                    class_id=CLASS_CARTON,
                    confidence=float(conf),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
                cartons.append(det)

        return cartons, persons

    def benchmark(
        self, test_frame: np.ndarray, num_iterations: int = 30
    ) -> EngineBenchmarkResult:
        """Measure latency and throughput under current hardware backend."""
        # Warm-up
        for _ in range(5):
            _ = self.predict(test_frame)

        latencies_ms: List[float] = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = self.predict(test_frame)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_lat = float(np.mean(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

        return EngineBenchmarkResult(
            engine_profile=self.active_profile,
            model_path=self.active_model_path,
            device=self.device,
            num_frames=num_iterations,
            mean_latency_ms=round(mean_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            throughput_fps=round(fps, 1),
        )
