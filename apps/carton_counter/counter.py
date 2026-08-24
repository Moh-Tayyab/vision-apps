"""Counting logic: single, multi-angle, multi-frame, and 3D angle detection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from detector import CartonDetector, Detection, DetectionResult


@dataclass
class CountResult:
    count: int
    confidence_avg: float
    method: str
    processing_time_ms: float
    detections_per_view: Optional[List[DetectionResult]] = None

    def to_dict(self) -> dict:
        result = {
            "count": self.count,
            "confidence_avg": round(self.confidence_avg, 4),
            "method": self.method,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }
        if self.detections_per_view:
            result["per_view"] = [d.to_dict() for d in self.detections_per_view]
        return result


@dataclass
class PalletAngle:
    pitch: float
    roll: float
    yaw: float
    homography: Optional[np.ndarray] = None
    is_valid: bool = True

    def to_dict(self) -> dict:
        return {
            "pitch_deg": round(self.pitch, 2),
            "roll_deg": round(self.roll, 2),
            "yaw_deg": round(self.yaw, 2),
            "is_valid": self.is_valid,
        }


class AngleDetector:
    """Estimate 3D pallet angle from a single view using OpenCV perspective transforms."""

    REFERENCE_POINTS = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
    )

    def detect_angle(self, image: np.ndarray, detections: List[Detection]) -> PalletAngle:
        if len(detections) < 4:
            return PalletAngle(pitch=0, roll=0, yaw=0, is_valid=False)

        boxes = np.array([d.bbox for d in detections])
        centers = np.column_stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2,
        ])

        hull = cv2.convexHull(centers.astype(np.float32))
        if len(hull) < 4:
            return PalletAngle(pitch=0, roll=0, yaw=0, is_valid=False)

        rect = cv2.minAreaRect(hull)
        box_pts = cv2.boxPoints(rect)
        box_pts = self._order_points(box_pts)

        try:
            H, _ = cv2.findHomography(self.REFERENCE_POINTS, box_pts)
            if H is None:
                return PalletAngle(pitch=0, roll=0, yaw=0, is_valid=False)
        except cv2.error:
            return PalletAngle(pitch=0, roll=0, yaw=0, is_valid=False)

        pitch, roll, yaw = self._decompose_homography(H, image.shape)
        return PalletAngle(pitch=pitch, roll=roll, yaw=yaw, homography=H, is_valid=True)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _decompose_homography(self, H: np.ndarray, img_shape: tuple) -> Tuple[float, float, float]:
        h, w = img_shape[:2]
        try:
            _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, np.eye(3))
            best_idx = 0
            best_z = -1
            for i, n in enumerate(Ns):
                if n[2] > best_z:
                    best_z = n[2]
                    best_idx = i

            R = Rs[best_idx]
            rvec, _ = cv2.Rodrigues(R)
            angles = np.degrees(rvec.flatten())
            pitch = float(angles[0]) if len(angles) > 0 else 0.0
            roll = float(angles[1]) if len(angles) > 1 else 0.0
            yaw = float(angles[2]) if len(angles) > 2 else 0.0
            return pitch, roll, yaw
        except cv2.error:
            return 0.0, 0.0, 0.0

    def apply_perspective_correction(
        self, image: np.ndarray, angle: PalletAngle
    ) -> Optional[np.ndarray]:
        if not angle.is_valid or angle.homography is None:
            return None
        h, w = image.shape[:2]
        target = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        try:
            H_inv = np.linalg.inv(angle.homography)
            return cv2.warpPerspective(image, H_inv, (w, h))
        except np.linalg.LinAlgError:
            return None


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class CartonCounter:
    """Carton counting engine with single, multi-angle, and multi-frame modes."""

    def __init__(self, detector: CartonDetector, iou_threshold: float = 0.3):
        self.detector = detector
        self.iou_threshold = iou_threshold
        self.angle_detector = AngleDetector()

    def count_single(self, image: np.ndarray) -> CountResult:
        start = time.perf_counter()
        result = self.detector.detect(image)
        elapsed_ms = (time.perf_counter() - start) * 1000

        count = len(result.detections)
        avg_conf = (
            np.mean([d.confidence for d in result.detections])
            if result.detections
            else 0.0
        )

        return CountResult(
            count=count,
            confidence_avg=float(avg_conf),
            method="single",
            processing_time_ms=elapsed_ms,
        )

    def count_multi_angle(self, images: List[np.ndarray]) -> CountResult:
        start = time.perf_counter()
        all_detections: List[Detection] = []
        per_view: List[DetectionResult] = []

        for img in images:
            det_result = self.detector.detect(img)
            all_detections.extend(det_result.detections)
            per_view.append(det_result)

        merged = self._cluster_detections(all_detections)
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_conf = (
            np.mean([d.confidence for d in merged]) if merged else 0.0
        )

        return CountResult(
            count=len(merged),
            confidence_avg=float(avg_conf),
            method="multi_angle",
            processing_time_ms=elapsed_ms,
            detections_per_view=per_view,
        )

    def count_multi_frame(self, frames: List[np.ndarray]) -> CountResult:
        start = time.perf_counter()
        frame_counts: List[int] = []

        for frame in frames:
            result = self.detector.detect(frame)
            frame_counts.append(len(result.detections))

        elapsed_ms = (time.perf_counter() - start) * 1000

        if not frame_counts:
            return CountResult(
                count=0, confidence_avg=0.0,
                method="multi_frame_voting", processing_time_ms=elapsed_ms,
            )

        median_count = int(np.median(frame_counts))
        return CountResult(
            count=median_count,
            confidence_avg=0.0,
            method="multi_frame_voting",
            processing_time_ms=elapsed_ms,
        )

    def _cluster_detections(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []

        boxes = np.array([d.bbox for d in detections])
        n = len(boxes)
        visited = [False] * n
        clusters: List[List[int]] = []

        for i in range(n):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            for j in range(i + 1, n):
                if visited[j]:
                    continue
                if _compute_iou(boxes[i], boxes[j]) > self.iou_threshold:
                    cluster.append(j)
                    visited[j] = True
            clusters.append(cluster)

        merged: List[Detection] = []
        for cluster_indices in clusters:
            cluster_dets = [detections[i] for i in cluster_indices]
            best = max(cluster_dets, key=lambda d: d.confidence)
            avg_bbox = np.mean([d.bbox for d in cluster_dets], axis=0).tolist()
            merged.append(
                Detection(
                    bbox=avg_bbox,
                    confidence=best.confidence,
                    class_id=best.class_id,
                    class_name=best.class_name,
                )
            )

        return merged

    def detect_pallet_angle(self, image: np.ndarray) -> PalletAngle:
        det_result = self.detector.detect(image)
        return self.angle_detector.detect_angle(image, det_result.detections)

    def get_info(self) -> dict:
        return {
            "iou_threshold": self.iou_threshold,
            "model_info": self.detector.get_model_info(),
        }
