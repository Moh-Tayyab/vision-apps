"""Carton detection for top camera view.

Uses YOLO for real-time carton detection from overhead camera.
"""

from __future__ import annotations

import base64
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests

DEFAULT_MODEL_URL = "https://detect.roboflow.com/carton-counter-demo/7"
DEFAULT_CONFIDENCE = 0.36


class DetectorError(RuntimeError):
    """Detection could not be completed."""


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
    if rel_area < 0.004 or rel_area > 0.55:
        return False

    aspect = w / h
    if aspect < 0.25 or aspect > 4.5:
        return False

    return w >= 25 and h >= 25


class CartonDetector:
    """YOLO-based carton detection for top camera view."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_url: Optional[str] = None,
        confidence: float = DEFAULT_CONFIDENCE,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY", "")
        self.model_url = (model_url or os.getenv("ROBOFLOW_MODEL_URL")
                          or DEFAULT_MODEL_URL).rstrip("/")
        self.confidence = confidence
        self.timeout = timeout
        self.last_inference_ms = 0.0

        if not self.api_key:
            raise DetectorError(
                "ROBOFLOW_API_KEY is not set; add it to the project .env file."
            )

    def detect(
        self, image: np.ndarray, confidence: Optional[float] = None
    ) -> Tuple[List[dict], float]:
        """Detect cartons. Returns (boxes, inference_ms)."""
        start = time.perf_counter()
        conf = self.confidence if confidence is None else confidence

        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise DetectorError("could not encode image for inference")

        try:
            response = requests.post(
                f"{self.model_url}?api_key={self.api_key}&confidence={conf}",
                data=base64.b64encode(buf.tobytes()).decode("ascii"),
                headers={"Content-Type": "text/plain"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if status == 401:
                raise DetectorError(
                    "Roboflow rejected the API key (401). Check ROBOFLOW_API_KEY."
                ) from exc
            raise DetectorError(f"Roboflow inference failed (HTTP {status}).") from exc
        except requests.RequestException as exc:
            raise DetectorError(f"Could not reach Roboflow: {exc}") from exc

        img_h, img_w = image.shape[:2]
        boxes = []
        for pred in payload.get("predictions", []):
            w, h = pred["width"], pred["height"]
            x1 = pred["x"] - w / 2
            y1 = pred["y"] - h / 2
            x2 = pred["x"] + w / 2
            y2 = pred["y"] + h / 2
            if _is_plausible_carton(x1, y1, x2, y2, img_w, img_h):
                boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": pred["confidence"]
                })

        self.last_inference_ms = (time.perf_counter() - start) * 1000.0
        return boxes, self.last_inference_ms

    def info(self) -> dict:
        return {
            "backend": "roboflow_cloud",
            "model_url": self.model_url,
            "confidence": self.confidence,
            "last_inference_ms": round(self.last_inference_ms, 1),
        }
