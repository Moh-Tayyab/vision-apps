"""Temporal Face Tracking & Classification Consensus Engine.

Solves:
1. Bounding-box flickering and spatial jitter across successive video frames.
2. Single-frame classification flicker (e.g. temporary bad lighting flipping status between AUTHORIZED and UNAUTHORIZED).
3. Consistent person tracking by assigning a unique persistent `track_id` to each face.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    return float(inter_area / max(1, union_area))


def _boxes_conflict(box1: List[int], box2: List[int], iou_thresh: float = 0.22) -> bool:
    """Check if two boxes significantly overlap or if one contains the other (e.g. face vs chest)."""
    if _compute_iou(box1, box2) >= iou_thresh:
        return True
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return False
    inter_area = (x2 - x1) * (y2 - y1)
    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return (inter_area / area1 > 0.35) or (inter_area / area2 > 0.35)


@dataclass
class TrackedPerson:
    track_id: int
    bbox: List[int]
    status: str = "unknown"
    matched_name: Optional[str] = None
    confidence: float = 0.0
    distance: Optional[float] = None
    liveness_score: float = 1.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    missed_frames: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=6))
    name_history: deque = field(default_factory=lambda: deque(maxlen=6))

    def update(self, det: dict, now: float, smooth_alpha: float = 0.70) -> None:
        raw_box = det.get("bbox", [0, 0, 0, 0])
        # Exponential moving average spatial smoothing
        self.bbox = [
            int(smooth_alpha * raw_box[0] + (1.0 - smooth_alpha) * self.bbox[0]),
            int(smooth_alpha * raw_box[1] + (1.0 - smooth_alpha) * self.bbox[1]),
            int(smooth_alpha * raw_box[2] + (1.0 - smooth_alpha) * self.bbox[2]),
            int(smooth_alpha * raw_box[3] + (1.0 - smooth_alpha) * self.bbox[3]),
        ]
        self.confidence = det.get("confidence", self.confidence)
        self.distance = det.get("distance", self.distance)
        self.liveness_score = det.get("liveness_score", self.liveness_score)
        self.last_seen = now
        self.missed_frames = 0

        status = det.get("status", "unknown")
        name = det.get("matched_name")

        # If a different authorized person is detected, reset history to switch identity immediately
        if str(status).lower() == "authorized" and name and name != "Unknown":
            if self.matched_name and str(self.matched_name).lower() != str(name).lower() and str(self.status).lower() == "authorized":
                self.history.clear()
                self.name_history.clear()
            self.history.append(status)
            self.name_history.append(name)
        else:
            self.history.append(status)

        # Majority temporal voting consensus
        if self.history:
            cnt = Counter(self.history)
            self.status = cnt.most_common(1)[0][0]

        if self.name_history:
            name_cnt = Counter(self.name_history)
            self.matched_name = name_cnt.most_common(1)[0][0]

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "status": self.status,
            "matched_name": self.matched_name,
            "confidence": round(self.confidence, 3),
            "distance": round(self.distance, 4) if self.distance is not None else None,
            "liveness_score": round(self.liveness_score, 3),
            "age_seconds": round(time.time() - self.first_seen, 1),
        }


class FaceTracker:
    """Multi-target spatial IoU tracker with temporal classification consensus."""

    def __init__(self, iou_threshold: float = 0.30, max_missed_frames: int = 2):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._next_track_id = 1
        self._tracks: Dict[int, TrackedPerson] = {}

    def update(self, detected_faces: List[dict], now: Optional[float] = None) -> List[dict]:
        """Match detected faces with active tracks and return smoothed tracked persons."""
        current_time = now or time.time()

        if not detected_faces:
            # Increment missed frames and purge stale tracks
            to_remove = []
            for tid, trk in self._tracks.items():
                trk.missed_frames += 1
                if trk.missed_frames > self.max_missed_frames:
                    to_remove.append(tid)
            for tid in to_remove:
                del self._tracks[tid]
            return []

        # Pairwise IoU cost matrix
        active_tids = list(self._tracks.keys())
        matched_detections = set()
        matched_tracks = set()

        if active_tids:
            for d_idx, det in enumerate(detected_faces):
                d_box = det.get("bbox", [0, 0, 0, 0])
                best_iou = 0.0
                best_tid = None

                for tid in active_tids:
                    if tid in matched_tracks:
                        continue
                    trk = self._tracks[tid]
                    iou = _compute_iou(d_box, trk.bbox)
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_tid = tid

                if best_tid is not None:
                    self._tracks[best_tid].update(det, current_time)
                    matched_detections.add(d_idx)
                    matched_tracks.add(best_tid)

        # Create new tracks for unmatched detections (only if they don't overlap with an existing track)
        for d_idx, det in enumerate(detected_faces):
            if d_idx not in matched_detections:
                d_box = det.get("bbox", [0, 0, 0, 0])
                # Suppress if this detection overlaps or conflicts with ANY existing track
                overlaps_existing = any(
                    _boxes_conflict(d_box, trk.bbox)
                    for trk in self._tracks.values()
                )
                if overlaps_existing:
                    continue

                tid = self._next_track_id
                self._next_track_id += 1
                trk = TrackedPerson(
                    track_id=tid,
                    bbox=list(d_box),
                    status=det.get("status", "unknown"),
                    matched_name=det.get("matched_name"),
                    confidence=det.get("confidence", 0.0),
                    distance=det.get("distance"),
                    liveness_score=det.get("liveness_score", 1.0),
                    first_seen=current_time,
                    last_seen=current_time,
                )
                trk.history.append(det.get("status", "unknown"))
                if det.get("matched_name"):
                    trk.name_history.append(det.get("matched_name"))
                self._tracks[tid] = trk

        # Purge stale tracks
        to_remove = set()
        for tid, trk in self._tracks.items():
            if tid not in matched_tracks and tid not in [self._next_track_id - 1]:
                trk.missed_frames += 1
                if trk.missed_frames > self.max_missed_frames:
                    to_remove.add(tid)

        for tid in to_remove:
            del self._tracks[tid]

        # Inter-track deduplication: If any 2 active tracks overlap or conflict, keep the stronger one
        active_list = sorted(
            self._tracks.values(),
            key=lambda t: (t.confidence, -t.missed_frames, len(t.history)),
            reverse=True,
        )
        kept_tids = set()
        final_tracks = []
        for trk in active_list:
            conflict = False
            for kept in final_tracks:
                if _boxes_conflict(trk.bbox, kept.bbox):
                    conflict = True
                    break
            if not conflict:
                final_tracks.append(trk)
                kept_tids.add(trk.track_id)

        # Enforce unique authorized identity per frame:
        # A single enrolled person cannot be at two different places at the exact same moment.
        # If 2 tracks claim the same authorized name, the one with better distance/confidence wins.
        claimed_names = set()
        for trk in final_tracks:
            if str(trk.status).lower() == "authorized" and trk.matched_name and trk.matched_name != "Unknown":
                if trk.matched_name in claimed_names:
                    trk.status = "unauthorized"
                    trk.matched_name = "Unknown"
                    trk.history.clear()
                    trk.name_history.clear()
                else:
                    claimed_names.add(trk.matched_name)

        # Remove superseded duplicate tracks from memory
        for tid in list(self._tracks.keys()):
            if tid not in kept_tids and tid not in to_remove:
                del self._tracks[tid]

        return [trk.to_dict() for trk in final_tracks if trk.missed_frames <= 1]
