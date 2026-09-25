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


def _boxes_conflict(box1: List[int], box2: List[int], iou_thresh: float = 0.20) -> bool:
    """Check if two boxes significantly overlap, intersect, or have close centers."""
    if _compute_iou(box1, box2) >= iou_thresh:
        return True
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 > x1 and y2 > y1:
        inter_area = (x2 - x1) * (y2 - y1)
        area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        min_area = min(area1, area2)
        # If 20% or more of either box area is shared, treat as conflicting duplicate
        if (inter_area / float(min_area)) > 0.20:
            return True

    # Center distance conflict: two boxes with centers within face radius cannot be separate persons
    c1_x = (box1[0] + box1[2]) / 2.0
    c1_y = (box1[1] + box1[3]) / 2.0
    c2_x = (box2[0] + box2[2]) / 2.0
    c2_y = (box2[1] + box2[3]) / 2.0
    dist = ((c1_x - c2_x)**2 + (c1_y - c2_y)**2)**0.5
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    max_dim = max(w1, h1, w2, h2)
    if dist < max_dim * 0.75:
        return True

    return False


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
    last_identified_time: float = 0.0
    missed_frames: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=6))
    name_history: deque = field(default_factory=lambda: deque(maxlen=6))

    def update(self, det: dict, now: float, smooth_alpha: float = 0.70) -> None:
        raw_box = det.get("bbox", [0, 0, 0, 0])
        # Responsive spatial tracking:
        # If user moves (center movement > 6px), snap quickly (alpha=0.95) so the box never lags behind!
        # If nearly static, smooth gently (alpha=0.75) for jitter-free stability.
        c_old_x = (self.bbox[0] + self.bbox[2]) / 2.0
        c_old_y = (self.bbox[1] + self.bbox[3]) / 2.0
        c_new_x = (raw_box[0] + raw_box[2]) / 2.0
        c_new_y = (raw_box[1] + raw_box[3]) / 2.0
        move_dist = ((c_new_x - c_old_x)**2 + (c_new_y - c_old_y)**2)**0.5
        effective_alpha = 0.95 if move_dist > 6.0 else 0.75

        self.bbox = [
            int(effective_alpha * raw_box[0] + (1.0 - effective_alpha) * self.bbox[0]),
            int(effective_alpha * raw_box[1] + (1.0 - effective_alpha) * self.bbox[1]),
            int(effective_alpha * raw_box[2] + (1.0 - effective_alpha) * self.bbox[2]),
            int(effective_alpha * raw_box[3] + (1.0 - effective_alpha) * self.bbox[3]),
        ]
        self.confidence = det.get("confidence", self.confidence)
        self.distance = det.get("distance", self.distance)
        self.liveness_score = det.get("liveness_score", self.liveness_score)
        self.last_seen = now
        self.missed_frames = 0

        status = det.get("status", "unknown")
        name = det.get("matched_name")

        # If positively identified as authorized, switch identity immediately
        if str(status).lower() == "authorized" and name and name != "Unknown":
            if self.status != "authorized" or (self.matched_name and str(self.matched_name).lower() != str(name).lower()):
                self.history.clear()
                self.name_history.clear()
            self.history.append(status)
            self.name_history.append(name)
            self.matched_name = name
            self.status = "authorized"
            self.last_identified_time = now
        else:
            self.history.append(status)
            # If this update came from a full recognition pass (not just track_cache), update last_identified_time
            if det.get("vector_engine") != "track_cache":
                self.last_identified_time = now

        # Majority temporal voting consensus
        if self.history and self.status != "authorized":
            cnt = Counter(self.history)
            self.status = cnt.most_common(1)[0][0]

        if self.name_history and self.status != "authorized":
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

    def __init__(self, iou_threshold: float = 0.20, max_missed_frames: int = 35):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._next_track_id = 1
        self._tracks: Dict[int, TrackedPerson] = {}

    def set_track_identity(self, track_id: int, status: str, matched_name: Optional[str], distance: Optional[float] = None) -> None:
        """Explicitly update the track identity once async recognition completes."""
        trk = self._tracks.get(track_id)
        if trk is not None:
            trk.status = status
            trk.matched_name = matched_name
            trk.distance = distance
            if status == "authorized" and matched_name:
                trk.history.clear()
                trk.name_history.clear()
                trk.history.append("authorized")
                trk.name_history.append(matched_name)

    def find_matching_track(self, box: List[int], iou_thresh: float = 0.20) -> Optional[TrackedPerson]:
        """Find an active track that spatially matches this box (for identity verification caching)."""
        best_trk = None
        best_iou = 0.0
        bx1, by1, bx2, by2 = box
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0

        for trk in self._tracks.values():
            iou = _compute_iou(box, trk.bbox)
            if iou > best_iou and iou >= iou_thresh:
                best_iou = iou
                best_trk = trk

        # Fallback to Euclidean center distance if slight motion shifted the box
        if best_trk is None:
            best_dist = 9999.0
            max_r = max(bx2 - bx1, by2 - by1) * 1.25
            for trk in self._tracks.values():
                tcx = (trk.bbox[0] + trk.bbox[2]) / 2.0
                tcy = (trk.bbox[1] + trk.bbox[3]) / 2.0
                dist = ((bcx - tcx)**2 + (bcy - tcy)**2) ** 0.5
                if dist < max_r and dist < best_dist:
                    best_dist = dist
                    best_trk = trk

        return best_trk

    def update(self, detected_faces: List[dict], now: Optional[float] = None) -> List[dict]:
        """Match detected faces with active tracks and return smoothed tracked persons."""
        current_time = now or time.time()

        if not detected_faces:
            # Increment missed frames and purge stale tracks, but DO NOT drop survivors immediately (prevents gaps)
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
                best_score = 0.0
                best_tid = None
                dcx = (d_box[0] + d_box[2]) / 2.0
                dcy = (d_box[1] + d_box[3]) / 2.0
                dw = d_box[2] - d_box[0]
                dh = d_box[3] - d_box[1]
                max_r = max(dw, dh) * 1.25

                for tid in active_tids:
                    if tid in matched_tracks:
                        continue
                    trk = self._tracks[tid]
                    iou = _compute_iou(d_box, trk.bbox)
                    tcx = (trk.bbox[0] + trk.bbox[2]) / 2.0
                    tcy = (trk.bbox[1] + trk.bbox[3]) / 2.0
                    cdist = ((dcx - tcx)**2 + (dcy - tcy)**2)**0.5

                    if iou >= self.iou_threshold:
                        if iou > best_score:
                            best_score = iou
                            best_tid = tid
                    elif cdist < max_r:
                        score = 1.0 - (cdist / max_r)
                        if score > best_score:
                            best_score = score
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
                    last_identified_time=current_time if det.get("vector_engine") != "track_cache" else 0.0,
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
        # The track with the lowest distance (best vector match) wins the authorized name.
        claimed_names = set()
        for trk in sorted(final_tracks, key=lambda t: (t.distance if t.distance is not None else 999.0)):
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

        return [trk.to_dict() for trk in final_tracks if trk.missed_frames == 0]
