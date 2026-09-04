"""Helmet detection engine: local YOLO or Roboflow cloud backend.

Class conventions supported:
- person  -> person box
- helmet  -> worn helmet
- head    -> bare head (no helmet)

Any dataset using these class names works with either backend
(e.g. dataperson/safety-helmet-dataset on Roboflow Universe).
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import requests

PERSON_CLASSES = {"person", "worker", "man", "woman", "human"}
HELMET_CLASSES = {"helmet", "hardhat", "hard hat", "hard-hat", "with_helmet", "with helmet", "safety helmet", "safety_helmet"}
HEAD_CLASSES = {"head", "bare head", "bare_head", "no-helmet", "no_helmet", "without_helmet", "without helmet", "no helmet", "no-hardhat", "no_hardhat", "no hardhat"}
CAP_CLASSES = {"cap", "hat", "baseball cap", "baseball_cap", "sun cap", "sports cap", "baseball-hat", "baseball_hat"}


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
    """Local YOLO. Works out of the box with custom best.pt or pretrained models."""

    def __init__(self, model_path: str, conf_threshold: float = 0.38, imgsz: int = 640):
        from ultralytics import YOLO

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.model = YOLO(model_path)
        self._class_names = self.model.names

    def detect_boxes(self, image: np.ndarray, confidence: Optional[float] = None) -> List[Box]:
        conf = confidence if confidence is not None else self.conf_threshold
        results = self.model(image, conf=conf, imgsz=self.imgsz, verbose=False)
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
            "imgsz": self.imgsz,
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
        img_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
        conf = confidence if confidence is not None else self.confidence
        url = f"{self.model_url}?api_key={self.api_key}&confidence={conf}"
        response = requests.post(
            url,
            data=img_b64,
            headers={"Content-Type": "text/plain"},
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
        conf = float(os.getenv("CONF_THRESHOLD", "0.38"))
        imgsz = int(os.getenv("IMGSZ", "640"))

        if backend == "roboflow":
            model_url = os.getenv("ROBOFLOW_MODEL_URL", "")
            api_key = os.getenv("ROBOFLOW_API_KEY", "")
            if not model_url or not api_key:
                raise ValueError("ROBOFLOW_MODEL_URL and ROBOFLOW_API_KEY required for cloud backend")
            self._detector: _BaseDetector = RoboflowHelmetDetector(model_url, api_key, conf)
        else:
            # Check candidate model paths (best.pt preferred)
            model_path = os.getenv("MODEL_PATH", "best.pt")
            if not os.path.isabs(model_path):
                base_dir = os.path.dirname(__file__)
                candidates = [
                    os.path.join(base_dir, model_path),
                    os.path.join(base_dir, "best.pt"),
                    os.path.join(base_dir, "yolov8m-hard-hat-detection.pt"),
                    os.path.join(base_dir, "helmet_yolo.pt"),
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        model_path = cand
                        break
            self._detector = LocalHelmetDetector(model_path, conf, imgsz=imgsz)
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
        caps = [b for b in raw if b.class_name in CAP_CLASSES]

        matched_helmets = set()
        matched_heads = set()
        persons: List[PersonStatus] = []

        # 1. Match against detected person bounding boxes
        for b in raw:
            if b.class_name not in PERSON_CLASSES:
                continue

            person_helmets = [h for h in helmets if b.contains(h)]
            person_heads = [hd for hd in heads if b.contains(hd)]
            person_caps = [c for c in caps if b.contains(c)]

            has_helmet = len(person_helmets) > 0
            has_head = len(person_heads) > 0
            has_cap = len(person_caps) > 0

            for h in person_helmets:
                matched_helmets.add(id(h))
            for hd in person_heads:
                matched_heads.add(id(hd))

            if has_helmet:
                status = "helmet"
            elif has_head or has_cap:
                status = "no_helmet"
            else:
                # If neither head nor helmet detected in box, but other heads/helmets exist in scene
                status = "no_helmet" if (helmets or heads) else "unknown"

            persons.append(PersonStatus([b.x1, b.y1, b.x2, b.y2], b.confidence, status))

        # 2. Add standalone / distant heads or helmets that were not bounded by a person box
        unmatched_heads = [hd for hd in heads if id(hd) not in matched_heads]
        unmatched_helmets = [h for h in helmets if id(h) not in matched_helmets]

        for hd in unmatched_heads:
            # Check if covered by an unmatched helmet
            covered = any(h.x1 <= hd.cx <= h.x2 and h.y1 <= hd.cy <= h.y2 for h in unmatched_helmets)
            persons.append(
                PersonStatus(
                    [hd.x1, hd.y1, hd.x2, hd.y2],
                    hd.confidence,
                    "helmet" if covered else "no_helmet",
                )
            )

        for h in unmatched_helmets:
            # If not already matched to an unmatched head
            if not any(h.x1 <= hd.cx <= h.x2 and h.y1 <= hd.cy <= h.y2 for hd in unmatched_heads):
                persons.append(
                    PersonStatus(
                        [h.x1, h.y1, h.x2, h.y2],
                        h.confidence,
                        "helmet",
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


def draw_helmet_detections(image: np.ndarray, result: FrameResult) -> np.ndarray:
    """Render bounding boxes, status tags, and top KPI banner onto the image."""
    vis = image.copy()
    h, w = vis.shape[:2]

    colors = {
        "helmet": (34, 197, 94),     # Green (#22c55e)
        "no_helmet": (0, 0, 239),     # Red (#ef4444)
        "unknown": (0, 165, 255),    # Orange
    }
    status_labels = {
        "helmet": "HELMET (SAFE)",
        "no_helmet": "NO HELMET (VIOLATION)",
        "unknown": "PERSON (UNKNOWN)",
    }

    safe_count = sum(1 for p in result.persons if p.status == "helmet")
    violation_count = sum(1 for p in result.persons if p.status == "no_helmet")
    total_count = len(result.persons)

    for p in result.persons:
        x1, y1, x2, y2 = [int(c) for c in p.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = colors.get(p.status, (255, 255, 255))

        # Main bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label tag above bounding box
        label = f"{status_labels.get(p.status, p.status)} {p.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y1 = max(0, y1 - th - 8)
        label_y2 = y1 if y1 >= th + 8 else y1 + th + 8
        cv2.rectangle(vis, (x1, label_y1), (x1 + tw + 6, label_y2), color, -1)
        cv2.putText(vis, label, (x1 + 3, label_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255) if p.status == "no_helmet" else (0, 0, 0), 1, cv2.LINE_AA)

    # Top KPI Banner Overlay
    overlay = vis.copy()
    banner_w = min(420, w - 20)
    banner_h = 50
    cv2.rectangle(overlay, (10, 10), (10 + banner_w, 10 + banner_h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)

    border_color = (0, 0, 239) if violation_count > 0 else ((34, 197, 94) if safe_count > 0 else (100, 116, 139))
    cv2.rectangle(vis, (10, 10), (10 + banner_w, 10 + banner_h), border_color, 2)

    status_text = f"PERSONS: {total_count}  |  SAFE: {safe_count}  |  VIOLATIONS: {violation_count}"
    cv2.putText(vis, status_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (248, 250, 252), 2, cv2.LINE_AA)
    return vis
