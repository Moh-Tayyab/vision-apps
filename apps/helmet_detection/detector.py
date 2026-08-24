"""Helmet detection engine: local YOLO or Roboflow cloud backend.

Class conventions supported:
- person  -> person box
- helmet  -> worn helmet
- head    -> bare head (no helmet)

Any dataset using these class names works with either backend
(e.g. dataperson/safety-helmet-dataset on Roboflow Universe).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import requests

PERSON_CLASSES = {"person"}
HELMET_CLASSES = {"helmet", "hardhat", "hard hat"}
HEAD_CLASSES = {"head", "bare head", "no-helmet"}


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def contains(self, other: "Box", top_frac: float = 0.7) -> bool:
        """True if ``other``'s center lies inside the top part of this box."""
        return (
            self.x1 <= other.cx <= self.x2
            and self.y1 <= other.cy <= self.y1 + (self.y2 - self.y1) * top_frac
        )


@dataclass
class PersonStatus:
    bbox: List[float]
    confidence: float
    status: str  # "helmet" | "no_helmet" | "unknown"

    def to_dict(self) -> dict:
        return {
            "bbox": [round(v, 1) for v in self.bbox],
            "confidence": round(self.confidence, 4),
            "status": self.status,
        }


@dataclass
class FrameResult:
    persons: List[PersonStatus]
    raw_boxes: List[Box] = field(default_factory=list)
    inference_time_ms: float = 0.0

    @property
    def violations(self) -> List[PersonStatus]:
        return [p for p in self.persons if p.status == "no_helmet"]

    def to_dict(self) -> dict:
        return {
            "persons": [p.to_dict() for p in self.persons],
            "num_persons": len(self.persons),
            "num_violations": len(self.violations),
            "inference_time_ms": round(self.inference_time_ms, 2),
        }


class _BaseDetector:
    def detect_boxes(self, image: np.ndarray, confidence: Optional[float] = None) -> List[Box]:
        raise NotImplementedError

    def get_model_info(self) -> dict:
        raise NotImplementedError


class LocalHelmetDetector(_BaseDetector):
    """Local YOLO. Works out of the box with COCO (persons only, status=unknown);
    full helmet/no-helmet when given weights trained on helmet/head classes."""

    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        from ultralytics import YOLO

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)
        self._class_names = self.model.names

    def detect_boxes(self, image: np.ndarray, confidence: Optional[float] = None) -> List[Box]:
        conf = confidence if confidence is not None else self.conf_threshold
        results = self.model(image, conf=conf, verbose=False)
        boxes: List[Box] = []
        for result in results:
            if result.boxes is None:
                continue
            for b in result.boxes:
                cls_name = self._class_names.get(int(b.cls[0]), str(int(b.cls[0])))
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append(Box(x1, y1, x2, y2, float(b.conf[0]), cls_name.lower()))
        return boxes

    def get_model_info(self) -> dict:
        return {
            "backend": "local_yolo",
            "model_path": self.model_path,
            "conf_threshold": self.conf_threshold,
            "classes": list(self._class_names.values()) if isinstance(self._class_names, dict) else self._class_names,
        }


class RoboflowHelmetDetector(_BaseDetector):
    """Roboflow-hosted helmet model via REST."""

    def __init__(self, model_url: str, api_key: str, confidence: float = 0.5):
        self.model_url = model_url.rstrip("/")
        self.api_key = api_key
        self.confidence = confidence

    def detect_boxes(self, image: np.ndarray, confidence: Optional[float] = None) -> List[Box]:
        start = time.perf_counter()
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        conf = confidence if confidence is not None else self.confidence
        url = f"{self.model_url}?api_key={self.api_key}&confidence={conf}"
        response = requests.post(
            url,
            data=buffer.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        boxes: List[Box] = []
        for pred in data.get("predictions", []):
            w, h = pred["width"], pred["height"]
            boxes.append(
                Box(
                    pred["x"] - w / 2,
                    pred["y"] - h / 2,
                    pred["x"] + w / 2,
                    pred["y"] + h / 2,
                    pred["confidence"],
                    pred.get("class", "unknown").lower(),
                )
            )
        return boxes

    def get_model_info(self) -> dict:
        return {
            "backend": "roboflow_cloud",
            "model_url": self.model_url,
            "confidence": self.confidence,
        }


class HelmetDetector:
    """Facade: picks backend from env and derives per-person helmet status."""

    def __init__(self):
        backend = os.getenv("MODEL_BACKEND", "local")
        conf = float(os.getenv("CONF_THRESHOLD", "0.5"))
        if backend == "roboflow":
            model_url = os.getenv("ROBOFLOW_MODEL_URL", "")
            api_key = os.getenv("ROBOFLOW_API_KEY", "")
            if not model_url or not api_key:
                raise ValueError("ROBOFLOW_MODEL_URL and ROBOFLOW_API_KEY required for cloud backend")
            self._detector: _BaseDetector = RoboflowHelmetDetector(model_url, api_key, conf)
        else:
            model_path = os.getenv("MODEL_PATH", "yolo11n.pt")
            self._detector = LocalHelmetDetector(model_path, conf)
        self._backend = backend

    @property
    def backend(self) -> str:
        return self._backend

    def detect(self, image: np.ndarray, confidence: Optional[float] = None) -> FrameResult:
        start = time.perf_counter()
        raw = self._detector.detect_boxes(image, confidence=confidence)
        elapsed_ms = (time.perf_counter() - start) * 1000

        helmets = [b for b in raw if b.class_name in HELMET_CLASSES]
        heads = [b for b in raw if b.class_name in HEAD_CLASSES]

        persons: List[PersonStatus] = []
        for b in raw:
            if b.class_name not in PERSON_CLASSES:
                continue
            has_helmet = any(b.contains(h) for h in helmets)
            has_head = any(b.contains(hd) for hd in heads)
            if has_helmet:
                status = "helmet"
            elif has_head:
                status = "no_helmet"
            else:
                status = "no_helmet" if (helmets or heads) else "unknown"
            persons.append(PersonStatus([b.x1, b.y1, b.x2, b.y2], b.confidence, status))

        # Datasets that label heads/helmets without person boxes still yield results.
        if not persons:
            for hd in heads:
                covered = any(h.x1 <= hd.cx <= h.x2 and h.y1 <= hd.cy <= h.y2 for h in helmets)
                persons.append(
                    PersonStatus(
                        [hd.x1, hd.y1, hd.x2, hd.y2],
                        hd.confidence,
                        "helmet" if covered else "no_helmet",
                    )
                )

        return FrameResult(persons=persons, raw_boxes=raw, inference_time_ms=elapsed_ms)

    def get_model_info(self) -> dict:
        info = self._detector.get_model_info()
        info["status_logic"] = {
            "helmet_classes": sorted(HELMET_CLASSES),
            "head_classes": sorted(HEAD_CLASSES),
            "person_classes": sorted(PERSON_CLASSES),
        }
        return info
