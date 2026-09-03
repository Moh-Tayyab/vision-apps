"""Carton detection for top camera view.

Uses local YOLO model (.pt) for real-time carton detection from overhead camera.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_MODEL_PATH = "best.pt"
DEFAULT_CONFIDENCE = 0.36
DEFAULT_NMS_IOU = 0.35
DEFAULT_IOS_THRESH = 0.58


class DetectorError(RuntimeError):
    """Detection could not be completed."""


def compute_iou(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> float:
    """Compute standard Intersection over Union (IoU)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_ios(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Smaller Area (containment score)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    min_area = min(area1, area2)
    return inter / min_area if min_area > 0 else 0.0


def non_max_suppression(
    boxes: List[dict],
    iou_thresh: float = DEFAULT_NMS_IOU,
    ios_thresh: float = DEFAULT_IOS_THRESH,
) -> List[dict]:
    """Applies IoU Non-Maximum Suppression and containment deduplication."""
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b.get("confidence", 0.0), reverse=True)
    kept: List[dict] = []

    for cur in sorted_boxes:
        cur_box = (cur["x1"], cur["y1"], cur["x2"], cur["y2"])
        suppressed = False
        for saved in kept:
            saved_box = (saved["x1"], saved["y1"], saved["x2"], saved["y2"])
            if compute_iou(cur_box, saved_box) > iou_thresh or compute_ios(cur_box, saved_box) > ios_thresh:
                suppressed = True
                break
        if not suppressed:
            kept.append(cur)

    return kept


def _is_plausible_carton(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> bool:
    """Reject boxes that cannot be a single carton face."""
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False

    img_area = img_w * img_h
    if img_area <= 0:
        return False

    rel_area = (w * h) / img_area
    if rel_area < 0.003 or rel_area > 0.65:
        return False

    aspect = w / h
    if aspect < 0.25 or aspect > 4.5:
        return False

    return w >= 20 and h >= 20


class CartonDetector:
    """Local YOLO-based carton detection for top camera view."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = DEFAULT_CONFIDENCE,
        nms_iou: float = DEFAULT_NMS_IOU,
        ios_thresh: float = DEFAULT_IOS_THRESH,
        device: Optional[str] = None,
    ):
        from ultralytics import YOLO

        self.model_path = model_path or os.getenv("YOLO_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.ios_thresh = ios_thresh
        self.last_inference_ms = 0.0
        self.device = device

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise DetectorError(
                f"YOLO model not found: {self.model_path}. "
                "Train your model and place the .pt file in the project root."
            )

        self.model = YOLO(str(model_file))
        if self.device:
            self.model.to(self.device)

    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        apply_nms: bool = True,
    ) -> Tuple[List[dict], float]:
        """Detect cartons. Returns (boxes, inference_ms)."""
        start = time.perf_counter()
        conf = self.confidence if confidence is None else confidence

        results = self.model(image, conf=conf, verbose=False)

        img_h, img_w = image.shape[:2]
        raw_boxes = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                conf_score = float(box.conf[0])

                if _is_plausible_carton(x1, y1, x2, y2, img_w, img_h):
                    raw_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "confidence": conf_score
                    })

        boxes = non_max_suppression(raw_boxes, self.nms_iou, self.ios_thresh) if apply_nms else raw_boxes

        self.last_inference_ms = (time.perf_counter() - start) * 1000.0
        return boxes, self.last_inference_ms

    def info(self) -> dict:
        return {
            "backend": "local_yolo",
            "model_path": self.model_path,
            "confidence": self.confidence,
            "nms_iou": self.nms_iou,
            "ios_thresh": self.ios_thresh,
            "last_inference_ms": round(self.last_inference_ms, 1),
        }
