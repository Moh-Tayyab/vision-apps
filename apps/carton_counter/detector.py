"""YOLO detection engine with local and Roboflow cloud backends."""

from __future__ import annotations

import os
import time
import base64
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import requests
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class DetectionResult:
    detections: List[Detection]
    inference_time_ms: float
    image_size: tuple[int, int]
    model_name: str

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "count": len(self.detections),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "image_size": list(self.image_size),
            "model_name": self.model_name,
        }


@dataclass
class InferenceStats:
    total_inferences: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    last_inference_time_ms: float = 0.0

    def update(self, time_ms: float) -> None:
        self.total_inferences += 1
        self.total_time_ms += time_ms
        self.last_inference_time_ms = time_ms
        self.avg_time_ms = self.total_time_ms / self.total_inferences


class LocalYOLODetector:
    """Local YOLO detection using ultralytics."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.5,
        target_classes: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes or ["carton", "box"]
        self.stats = InferenceStats()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = YOLO(model_path)
        self._class_names = self.model.names

    def detect(self, image: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stats.update(elapsed_ms)

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self._class_names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=float(box.conf[0]),
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        return DetectionResult(
            detections=detections,
            inference_time_ms=elapsed_ms,
            image_size=(image.shape[1], image.shape[0]),
            model_name=os.path.basename(self.model_path),
        )

    def get_model_info(self) -> dict:
        return {
            "backend": "local_yolo",
            "model_path": self.model_path,
            "conf_threshold": self.conf_threshold,
            "target_classes": self.target_classes,
            "class_names": self._class_names,
            "stats": {
                "total_inferences": self.stats.total_inferences,
                "avg_time_ms": round(self.stats.avg_time_ms, 2),
                "last_time_ms": round(self.stats.last_inference_time_ms, 2),
            },
        }


class RoboflowCloudDetector:
    """Roboflow cloud inference via REST API."""

    def __init__(
        self,
        model_url: str,
        api_key: str,
        confidence: float = 0.5,
    ):
        self.model_url = model_url.rstrip("/")
        self.api_key = api_key
        self.confidence = confidence
        self.stats = InferenceStats()
        self._class_names: dict[int, str] = {}

    def detect(self, image: np.ndarray) -> DetectionResult:
        start = time.perf_counter()

        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_bytes = buffer.tobytes()

        url = f"{self.model_url}?api_key={self.api_key}&confidence={self.confidence}"
        response = requests.post(
            url,
            data=img_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stats.update(elapsed_ms)

        detections: List[Detection] = []
        for pred in data.get("predictions", []):
            x = pred["x"]
            y = pred["y"]
            w = pred["width"]
            h = pred["height"]
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            cls_name = pred.get("class", "unknown")
            cls_id = pred.get("class_id", -1)

            self._class_names[cls_id] = cls_name
            detections.append(
                Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=pred["confidence"],
                    class_id=cls_id,
                    class_name=cls_name,
                )
            )

        h_img, w_img = image.shape[:2]
        return DetectionResult(
            detections=detections,
            inference_time_ms=elapsed_ms,
            image_size=(w_img, h_img),
            model_name=data.get("model", "roboflow_cloud"),
        )

    def get_model_info(self) -> dict:
        return {
            "backend": "roboflow_cloud",
            "model_url": self.model_url,
            "confidence": self.confidence,
            "stats": {
                "total_inferences": self.stats.total_inferences,
                "avg_time_ms": round(self.stats.avg_time_ms, 2),
                "last_time_ms": round(self.stats.last_inference_time_ms, 2),
            },
        }


class CartonDetector:
    """Facade that switches between local and cloud backends via env vars."""

    def __init__(self):
        backend = os.getenv("MODEL_BACKEND", "local")
        conf = float(os.getenv("CONF_THRESHOLD", "0.5"))

        if backend == "roboflow":
            model_url = os.getenv("ROBOFLOW_MODEL_URL", "")
            api_key = os.getenv("ROBOFLOW_API_KEY", "")
            if not model_url or not api_key:
                raise ValueError("ROBOFLOW_MODEL_URL and ROBOFLOW_API_KEY required for cloud backend")
            self._detector: LocalYOLODetector | RoboflowCloudDetector = RoboflowCloudDetector(
                model_url=model_url, api_key=api_key, confidence=conf
            )
        else:
            model_path = os.getenv("MODEL_PATH", "yolo11n.pt")
            self._detector = LocalYOLODetector(model_path=model_path, conf_threshold=conf)

        self._backend = backend

    def detect(self, image: np.ndarray) -> DetectionResult:
        return self._detector.detect(image)

    def get_model_info(self) -> dict:
        return self._detector.get_model_info()

    @property
    def backend(self) -> str:
        return self._backend
