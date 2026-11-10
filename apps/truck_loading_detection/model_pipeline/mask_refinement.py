"""
Production-Grade Carton Instance Segmentation & Mask Refinement Engine.

Provides pixel-accurate carton boundary masks, worker-occlusion disentanglement,
temporal contour stabilization, and pickup/carrying state tracking.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RefinedCartonMask:
    """Pixel-accurate carton instance mask with geometric and temporal metadata."""

    polygon_pts: np.ndarray  # (N, 2) float32 coordinates
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    area_px: float
    confidence: float
    is_occluded_by_worker: bool = False
    handling_state: str = "UNKNOWN"  # AT_REST, PICKUP_LIFTING, BEING_CARRIED, OCCLUDED, UNKNOWN
    worker_overlap_ratio: float = 0.0
    raw_polygon: Optional[np.ndarray] = None

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class CartonMaskRefiner:
    """
    Refines raw YOLO segmentation polygon masks for warehouse production:
    1. Worker-Carton Disentanglement: Subtracts overlapping worker limbs/clothing from carton masks.
    2. Contour Smoothing & Regularization: Removes jagged raster noise without losing sharp box corners.
    3. Temporal Stability: Blends consecutive frame contours to eliminate flickering (with perimeter resampling).
    4. Motion & Handling State Assessment: Multi-evidence kinematics evaluator (AT_REST, PICKUP_LIFTING, BEING_CARRIED, OCCLUDED, UNKNOWN).
    """

    def __init__(
        self,
        smoothing_epsilon_ratio: float = 0.003,
        worker_subtraction_margin_px: int = 2,
        temporal_blend_alpha: float = 0.65,
        min_carton_area_px: float = 150.0,
    ) -> None:
        self.smoothing_epsilon_ratio = float(smoothing_epsilon_ratio)
        self.worker_subtraction_margin_px = int(worker_subtraction_margin_px)
        self.temporal_blend_alpha = float(temporal_blend_alpha)
        self.min_carton_area_px = float(min_carton_area_px)

        # Track history for temporal contour stabilization: track_id -> previous polygon_pts
        self._previous_masks: Dict[str, np.ndarray] = {}
        # Kinematic state history: track_id -> dict
        self._kinematic_history: Dict[str, Dict[str, Any]] = {}

    def reset(self) -> None:
        self._previous_masks.clear()
        self._kinematic_history.clear()

    @staticmethod
    def _resample_polygon(pts: np.ndarray, n_points: int = 16) -> np.ndarray:
        """Resamples polygon vertices along its perimeter to a fixed number of equidistant points."""
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 3:
            return pts
        # Compute cumulative distance along perimeter
        dists = np.sqrt(np.sum(np.diff(np.vstack([pts, pts[0]]), axis=0) ** 2, axis=1))
        cum_dists = np.insert(np.cumsum(dists), 0, 0.0)
        total_len = cum_dists[-1]
        if total_len <= 1e-4:
            return pts[:n_points]
        target_dists = np.linspace(0, total_len, n_points, endpoint=False)
        resampled = np.zeros((n_points, 2), dtype=np.float32)
        for i, td in enumerate(target_dists):
            idx = int(np.searchsorted(cum_dists, td, side="right") - 1)
            idx = max(0, min(len(pts) - 1, idx))
            next_idx = (idx + 1) % len(pts)
            seg_len = dists[idx]
            if seg_len > 1e-4:
                frac = (td - cum_dists[idx]) / seg_len
                resampled[i] = pts[idx] + frac * (pts[next_idx] - pts[idx])
            else:
                resampled[i] = pts[idx]
        return resampled

    def refine_carton_mask(
        self,
        raw_polygon: Union[np.ndarray, Sequence[Tuple[float, float]]],
        image_shape: Tuple[int, int],
        worker_polygons: Optional[Sequence[Union[np.ndarray, Sequence[Tuple[float, float]]]]] = None,
        confidence: float = 1.0,
        track_id: Optional[str] = None,
        timestamp_sec: float = 0.0,
    ) -> Optional[RefinedCartonMask]:
        """
        Refines a single carton polygon mask, disentangling worker overlap and smoothing boundaries.
        """
        h, w = image_shape[:2]
        if raw_polygon is None or len(raw_polygon) < 3:
            return None

        poly = np.array(raw_polygon, dtype=np.float32)
        if poly.ndim != 2 or poly.shape[1] != 2:
            return None

        raw_poly_copy = poly.copy()

        # 1. Rasterize Carton Mask into Binary Buffer
        carton_bin = np.zeros((h, w), dtype=np.uint8)
        poly_int = np.clip(np.round(poly).astype(np.int32), 0, [w - 1, h - 1])
        cv2.fillPoly(carton_bin, [poly_int], 255)

        initial_area = float(cv2.countNonZero(carton_bin))
        if initial_area < self.min_carton_area_px:
            return None

        # 2. Worker-Carton Disentanglement
        is_occluded = False
        worker_overlap_ratio = 0.0

        if worker_polygons:
            worker_bin = np.zeros((h, w), dtype=np.uint8)
            for w_poly in worker_polygons:
                if w_poly is not None and len(w_poly) >= 3:
                    wp_int = np.clip(np.round(np.array(w_poly)).astype(np.int32), 0, [w - 1, h - 1])
                    cv2.fillPoly(worker_bin, [wp_int], 255)

            # Measure overlap (intersection)
            overlap_bin = cv2.bitwise_and(carton_bin, worker_bin)
            overlap_px = float(cv2.countNonZero(overlap_bin))

            if overlap_px > 0:
                worker_overlap_ratio = overlap_px / max(1.0, initial_area)
                # If worker hands or body cover part of the carton (up to 70% overlap):
                if 0.02 < worker_overlap_ratio < 0.70:
                    is_occluded = True
                    if self.worker_subtraction_margin_px > 0:
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (self.worker_subtraction_margin_px * 2 + 1, self.worker_subtraction_margin_px * 2 + 1)
                        )
                        dilated_worker = cv2.dilate(worker_bin, kernel)
                    else:
                        dilated_worker = worker_bin

                    cleaned_carton = cv2.subtract(carton_bin, dilated_worker)
                    cleaned_area = float(cv2.countNonZero(cleaned_carton))
                    # Only accept subtraction if it preserves >= 40% of the visible carton body
                    if cleaned_area >= max(self.min_carton_area_px, 0.40 * initial_area):
                        carton_bin = cleaned_carton
                elif worker_overlap_ratio >= 0.70:
                    is_occluded = True
                    # If confidence is low (< 0.25) and almost entirely enveloped by worker body,
                    # reject as worker clothing texture / false phantom detection
                    if confidence < 0.25:
                        independent_carton = cv2.subtract(carton_bin, worker_bin)
                        independent_area = float(cv2.countNonZero(independent_carton))
                        if independent_area < max(200.0, 0.10 * initial_area):
                            return None
                    # Otherwise, worker is holding carton with heavy occlusion: preserve visible carton

        # 3. Extract Smooth Outer Contour
        contours, _ = cv2.findContours(carton_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Select the dominant carton contour
        primary_contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(primary_contour))
        if area < self.min_carton_area_px:
            return None

        # Ramer-Douglas-Peucker polygon approximation to preserve sharp box corners
        perimeter = cv2.arcLength(primary_contour, True)
        epsilon = max(1.0, perimeter * self.smoothing_epsilon_ratio)
        approx = cv2.approxPolyDP(primary_contour, epsilon, True)
        if len(approx) < 3:
            approx = primary_contour

        smooth_pts = approx.reshape(-1, 2).astype(np.float32)

        # 4. Temporal Stabilization (Filter out high-frequency polygon flicker across frames)
        if track_id and track_id in self._previous_masks:
            prev_pts = self._previous_masks[track_id]
            if len(prev_pts) == len(smooth_pts):
                # Smooth vertices with exponential moving average
                smooth_pts = (self.temporal_blend_alpha * smooth_pts) + (
                    (1.0 - self.temporal_blend_alpha) * prev_pts
                )
            else:
                # Common perimeter resampling when vertex counts differ
                try:
                    resampled_prev = self._resample_polygon(prev_pts, n_points=16)
                    resampled_curr = self._resample_polygon(smooth_pts, n_points=16)
                    smooth_pts = (self.temporal_blend_alpha * resampled_curr) + (
                        (1.0 - self.temporal_blend_alpha) * resampled_prev
                    )
                except Exception:
                    pass

        if track_id:
            self._previous_masks[track_id] = smooth_pts.copy()

        # 5. Compute Tight Bounding Box
        bx, by, bw, bh = cv2.boundingRect(smooth_pts.astype(np.int32))
        bbox = (float(bx), float(by), float(bx + bw), float(by + bh))

        # 6. Multi-Evidence Handling State Assessment (Phases 3 & 5)
        handling_state = "UNKNOWN"
        curr_c = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

        if track_id is not None:
            if track_id not in self._kinematic_history:
                self._kinematic_history[track_id] = {
                    "first_center": curr_c,
                    "last_center": curr_c,
                    "last_time": timestamp_sec,
                    "recent_speeds": deque(maxlen=5),
                    "consecutive_moving": 0,
                    "consecutive_still": 0,
                    "current_state": "AT_REST",
                }
                handling_state = "OCCLUDED" if is_occluded else "AT_REST"
            else:
                kh = self._kinematic_history[track_id]
                px, py = kh["last_center"]
                pt = kh["last_time"]
                dt = max(1e-4, timestamp_sec - pt)
                inst_speed = float(np.hypot(curr_c[0] - px, curr_c[1] - py) / dt)
                kh["recent_speeds"].append(inst_speed)
                smoothed_speed = float(np.mean(kh["recent_speeds"]))

                first_c = kh["first_center"]
                disp_from_init = float(np.hypot(curr_c[0] - first_c[0], curr_c[1] - first_c[1]))
                vy = float((curr_c[1] - py) / dt)  # negative vy is upward elevation

                if smoothed_speed >= 8.0:
                    kh["consecutive_moving"] += 1
                    kh["consecutive_still"] = 0
                else:
                    kh["consecutive_still"] += 1
                    kh["consecutive_moving"] = 0

                prev_state = kh["current_state"]

                # DECISION RULES (Independent of worker proximity alone):
                # 1. Sustained motion with displacement -> BEING_CARRIED
                if kh["consecutive_moving"] >= 3 and disp_from_init >= 20.0 and smoothed_speed >= 12.0:
                    handling_state = "BEING_CARRIED"
                # 2. Initial upward elevation or jump from rest -> PICKUP_LIFTING
                elif (vy < -6.0 or (smoothed_speed >= 9.0 and kh["consecutive_moving"] <= 3)) and prev_state in ("AT_REST", "OCCLUDED", "UNKNOWN"):
                    handling_state = "PICKUP_LIFTING"
                # 3. Transition from carried/lifting back to rest -> PLACEMENT_RELEASE / AT_REST
                elif kh["consecutive_still"] >= 4 and prev_state in ("BEING_CARRIED", "PICKUP_LIFTING"):
                    handling_state = "AT_REST"
                    # Reset rest anchor to new resting position
                    kh["first_center"] = curr_c
                # 4. Stationary but worker overlaps -> OCCLUDED (NOT carried!)
                elif is_occluded and smoothed_speed < 8.0 and disp_from_init < 15.0:
                    handling_state = "OCCLUDED"
                # 5. Stationary without overlap -> AT_REST
                elif smoothed_speed < 8.0 and disp_from_init < 15.0:
                    handling_state = "AT_REST"
                # 6. Moderate movement without worker -> preserve or UNKNOWN
                elif smoothed_speed >= 8.0:
                    handling_state = prev_state if prev_state in ("PICKUP_LIFTING", "BEING_CARRIED") else "PICKUP_LIFTING"
                else:
                    handling_state = "UNKNOWN"

                kh["last_center"] = curr_c
                kh["last_time"] = timestamp_sec
                kh["current_state"] = handling_state

        return RefinedCartonMask(
            polygon_pts=smooth_pts,
            bbox=bbox,
            area_px=area,
            confidence=float(confidence),
            is_occluded_by_worker=is_occluded,
            handling_state=handling_state,
            worker_overlap_ratio=float(worker_overlap_ratio),
            raw_polygon=raw_poly_copy,
        )

    def disentangle_batch(
        self,
        carton_detections: Sequence[Any],
        worker_detections: Sequence[Any],
        image_shape: Tuple[int, int],
        timestamp_sec: float = 0.0,
    ) -> List[RefinedCartonMask]:
        """
        Disentangles all carton detections in a frame against all worker detections.
        """
        worker_polys = [
            getattr(w, "mask_polygon", None) or getattr(w, "polygon", None)
            for w in worker_detections
            if getattr(w, "mask_polygon", None) is not None or getattr(w, "polygon", None) is not None
        ]

        # Fallback to worker bboxes if polygon masks are not present on workers
        if not worker_polys:
            for w in worker_detections:
                bbox = getattr(w, "bbox", None)
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    worker_polys.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        refined_list: List[RefinedCartonMask] = []
        for det in carton_detections:
            poly = getattr(det, "mask_polygon", None)
            if poly is None:
                bbox = getattr(det, "bbox", None)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                else:
                    continue

            tid = getattr(det, "local_track_id", None) or getattr(det, "track_id", None)
            conf = getattr(det, "confidence", 1.0)
            res = self.refine_carton_mask(
                raw_polygon=poly,
                image_shape=image_shape,
                worker_polygons=worker_polys,
                confidence=conf,
                track_id=str(tid) if tid is not None else None,
                timestamp_sec=timestamp_sec,
            )
            if res is not None:
                refined_list.append(res)

        return refined_list
