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

        resolved = self._resolve_weights(model_path)
        self.model = YOLO(resolved)
        self._class_names = self.model.names

    @staticmethod
    def _resolve_weights(model_path: str) -> str:
        """Use an existing path, else fall back to a bare name ultralytics can auto-download."""
        if os.path.exists(model_path):
            return model_path
        basename = os.path.basename(model_path)
        if basename != model_path and os.path.exists(basename):
            return basename
        import re

        if re.fullmatch(r"yolo(v\d+)?[a-z0-9]*(-cls|-pose|-seg|-obb)?[nslmx](-pt)?\.pt", basename):
            return basename
        raise FileNotFoundError(
            f"Model not found: {model_path} (auto-download only supported for standard YOLO weight names)"
        )

    def detect(self, image: np.ndarray, confidence: Optional[float] = None) -> DetectionResult:
        start = time.perf_counter()
        results = self.model(image, conf=confidence if confidence is not None else self.conf_threshold, verbose=False)
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

    def detect(self, image: np.ndarray, confidence: Optional[float] = None) -> DetectionResult:
        start = time.perf_counter()

        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_bytes = buffer.tobytes()

        conf = confidence if confidence is not None else self.confidence
        url = f"{self.model_url}?api_key={self.api_key}&confidence={conf}"
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


class RoboflowWorkflowDetector:
    """Roboflow Workflow backend (e.g. SAHI sliced inference) via REST API.

    Runs a saved workspace workflow such as
    Image Slicer -> ObjectDetectionModel -> Detections Stitch and applies the
    confidence threshold client-side on the merged (stitched) predictions.
    """

    def __init__(
        self,
        workspace: str,
        workflow_id: str,
        api_key: str,
        api_url: str = "https://serverless.roboflow.com",
        confidence: float = 0.11,
        output_field: str = "stitched_predictions",
    ):
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.confidence = confidence
        self.output_field = output_field
        self.stats = InferenceStats()
        self._class_names: dict[int, str] = {}

    def detect(self, image: np.ndarray, confidence: Optional[float] = None) -> DetectionResult:
        start = time.perf_counter()

        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")

        url = f"{self.api_url}/{self.workspace}/workflows/{self.workflow_id}"
        payload = {
            "api_key": self.api_key,
            "inputs": {"image": img_b64},
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stats.update(elapsed_ms)

        # Response shape: "outputs": [{"<output_name>": {"image": ..., "predictions": [...]}}]
        outputs = data.get("outputs", [])
        field_obj = None
        for entry in outputs:
            if isinstance(entry, dict) and self.output_field in entry:
                field_obj = entry[self.output_field]
                break
        if field_obj is None and outputs and isinstance(outputs[0], dict) and outputs[0]:
            field_obj = next(iter(outputs[0].values()))
        raw_preds = (field_obj or {}).get("predictions", []) if isinstance(field_obj, dict) else []

        threshold = confidence if confidence is not None else self.confidence
        detections: List[Detection] = []
        for pred in raw_preds:
            if pred["confidence"] < threshold:
                continue
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            cls_name = pred.get("class", "unknown")
            cls_id = pred.get("class_id", -1)
            self._class_names[cls_id] = cls_name
            detections.append(
                Detection(
                    bbox=[x - w / 2, y - h / 2, x + w / 2, y + h / 2],
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
            model_name=f"workflow:{self.workspace}/{self.workflow_id}",
        )

    def get_model_info(self) -> dict:
        return {
            "backend": "roboflow_workflow",
            "workspace": self.workspace,
            "workflow": self.workflow_id,
            "confidence": self.confidence,
            "output_field": self.output_field,
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
            self._detector: LocalYOLODetector | RoboflowCloudDetector | RoboflowWorkflowDetector = (
                RoboflowCloudDetector(model_url=model_url, api_key=api_key, confidence=conf)
            )
        elif backend == "roboflow_workflow":
            workspace = os.getenv("ROBOFLOW_WORKSPACE", "")
            workflow_id = os.getenv("ROBOFLOW_WORKFLOW", "")
            api_key = os.getenv("ROBOFLOW_API_KEY", "")
            if not (workspace and workflow_id and api_key):
                raise ValueError(
                    "ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW and ROBOFLOW_API_KEY "
                    "required for roboflow_workflow backend"
                )
            self._detector = RoboflowWorkflowDetector(
                workspace=workspace,
                workflow_id=workflow_id,
                api_key=api_key,
                api_url=os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com"),
                confidence=conf,
                output_field=os.getenv("ROBOFLOW_WORKFLOW_OUTPUT", "stitched_predictions"),
            )
        else:
            model_path = os.getenv("MODEL_PATH", "models/yolo26m.pt")
            self._detector = LocalYOLODetector(model_path=model_path, conf_threshold=conf)

        self._backend = backend

    def detect(self, image: np.ndarray, confidence: Optional[float] = None) -> DetectionResult:
        return self._detector.detect(image, confidence=confidence)

    def get_model_info(self) -> dict:
        return self._detector.get_model_info()

    @property
    def backend(self) -> str:
        return self._backend
