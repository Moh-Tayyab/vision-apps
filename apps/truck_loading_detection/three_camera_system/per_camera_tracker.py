"""
Per-Camera Carton Tracklet Manager (Phase 2).

Manages 2D tracklets independently for each camera feed.
Maintains persistent track state, handles temporary occlusions / missed frames,
computes 2D velocity vectors, and assigns locally scoped unique track IDs.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .detector_adapter import (
    CanonicalCartonDetection,
)


class TrackStatus(str, enum.Enum):
    TENTATIVE = "TENTATIVE"       # Initial frames, not yet confirmed
    CONFIRMED = "CONFIRMED"       # Stable tracked carton
    OCCLUDED = "OCCLUDED"         # Temporarily missed detection (coasting)
    TERMINATED = "TERMINATED"     # Lost for > max_invisible_frames


@dataclass
class CartonTracklet:
    """Represents a single carton's motion history in one camera view."""

    camera_id: str
    local_track_id: str
    first_seen_frame: int
    first_seen_timestamp: float
    last_seen_frame: int
    last_seen_timestamp: float
    current_bbox: Tuple[float, float, float, float]
    current_confidence: float
    status: TrackStatus = TrackStatus.TENTATIVE
    mask_polygon: Optional[List[Tuple[float, float]]] = None
    trajectory: deque = field(default_factory=lambda: deque(maxlen=60))  # (cx, cy, timestamp_sec)
    total_visible_frames: int = 1
    consecutive_invisible_frames: int = 0
    # Verified observation state (Phase 2):
    last_observed_bbox: Tuple[float, float, float, float] = field(init=False)
    last_observed_polygon: Optional[List[Tuple[float, float]]] = field(init=False, default=None)
    last_observed_velocity: Tuple[float, float] = field(init=False, default=(0.0, 0.0))
    observed_trajectory: deque = field(init=False)

    def __post_init__(self) -> None:
        self.last_observed_bbox = self.current_bbox
        self.last_observed_polygon = [pt for pt in self.mask_polygon] if self.mask_polygon else None
        self.observed_trajectory = deque(maxlen=60)
        c = self.center
        self.observed_trajectory.append((c[0], c[1], self.last_seen_timestamp))

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.current_bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def velocity(self) -> Tuple[float, float]:
        """Estimated 2D velocity (vx, vy) in pixels/second over verified observed detections."""
        if hasattr(self, "observed_trajectory") and len(self.observed_trajectory) >= 2:
            p_first = self.observed_trajectory[0]
            p_last = self.observed_trajectory[-1]
            dt = p_last[2] - p_first[2]
            if dt > 1e-4:
                vx = (p_last[0] - p_first[0]) / dt
                vy = (p_last[1] - p_first[1]) / dt
                return (vx, vy)
        elif len(self.trajectory) >= 2:
            p_first = self.trajectory[0]
            p_last = self.trajectory[-1]
            dt = p_last[2] - p_first[2]
            if dt > 1e-4:
                vx = (p_last[0] - p_first[0]) / dt
                vy = (p_last[1] - p_first[1]) / dt
                return (vx, vy)
        return (0.0, 0.0)

    @property
    def displacement(self) -> float:
        """Euclidean distance traveled from initial detection to current position."""
        if not self.trajectory:
            return 0.0
        p0 = self.trajectory[0]
        curr = self.center
        return math.hypot(curr[0] - p0[0], curr[1] - p0[1])


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Intersection over Union for two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class PerCameraCartonTracker:
    """
    Independent 2D tracker for a single camera perspective.
    
    Guarantees:
    - Track IDs are scoped locally with camera prefix (e.g. 'cam_top_t0001').
    - Never shares IDs with another camera.
    - Handles temporary occlusion via coasting up to `max_invisible_frames`.
    - Updates velocity and spatial trajectories for crossing verification.
    """

    def __init__(
        self,
        camera_id: str,
        min_hits_to_confirm: int = 2,
        max_invisible_frames: int = 15,
        iou_threshold: float = 0.25,
        max_center_dist_px: float = 120.0,
    ) -> None:
        self.camera_id = str(camera_id)
        self.min_hits = int(min_hits_to_confirm)
        self.max_invisible = int(max_invisible_frames)
        self.iou_threshold = float(iou_threshold)
        self.max_dist = float(max_center_dist_px)

        self._next_local_id: int = 1
        self._active_tracklets: Dict[str, CartonTracklet] = {}
        self._terminated_tracklets: List[CartonTracklet] = []

    def reset(self) -> None:
        self._next_local_id = 1
        self._active_tracklets.clear()
        self._terminated_tracklets.clear()

    @property
    def active_tracklets(self) -> List[CartonTracklet]:
        return list(self._active_tracklets.values())

    def update(
        self,
        detections: List[CanonicalCartonDetection],
        frame_idx: int,
        timestamp_sec: float,
    ) -> List[CartonTracklet]:
        """
        Ingest current frame detections, update existing tracklets, create new ones,
        and manage coasting / termination.
        """
        active_ids = list(self._active_tracklets.keys())
        unmatched_dets = set(range(len(detections)))
        matched_tracks = set()

        # Build cost / similarity matrix: IoU with fallback to center distance
        if active_ids and detections:
            candidates: List[Tuple[float, str, int]] = []
            for tid in active_ids:
                trk = self._active_tracklets[tid]
                tcx, tcy = trk.center
                for did, det in enumerate(detections):
                    iou = compute_iou(trk.current_bbox, det.bbox)
                    dcx, dcy = det.center
                    dist = math.hypot(tcx - dcx, tcy - dcy)
                    
                    effective_dist = max(self.max_dist, 160.0) if trk.status == TrackStatus.OCCLUDED else self.max_dist
                    if iou >= self.iou_threshold or dist <= effective_dist:
                        # Priority score: higher IoU is better, lower dist is better
                        score = iou + max(0.0, 1.0 - (dist / effective_dist))
                        candidates.append((score, tid, did))

            # Greedy assignment by score descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            for score, tid, did in candidates:
                if tid in matched_tracks or did not in unmatched_dets:
                    continue

                # Matched! Update tracklet
                trk = self._active_tracklets[tid]
                det = detections[did]
                trk.last_seen_frame = frame_idx
                trk.last_seen_timestamp = timestamp_sec
                trk.current_bbox = det.bbox
                trk.last_observed_bbox = det.bbox
                trk.current_confidence = det.confidence
                trk.mask_polygon = det.mask_polygon
                trk.last_observed_polygon = [pt for pt in det.mask_polygon] if det.mask_polygon else None
                trk.total_visible_frames += 1
                trk.consecutive_invisible_frames = 0
                trk.observed_trajectory.append((det.center[0], det.center[1], timestamp_sec))
                trk.last_observed_velocity = trk.velocity
                trk.trajectory.append((det.center[0], det.center[1], timestamp_sec))

                if trk.status == TrackStatus.TENTATIVE and trk.total_visible_frames >= self.min_hits:
                    trk.status = TrackStatus.CONFIRMED
                elif trk.status == TrackStatus.OCCLUDED:
                    trk.status = TrackStatus.CONFIRMED

                matched_tracks.add(tid)
                unmatched_dets.remove(did)

        # Coast or terminate unmatched active tracklets with mathematically sound constant-velocity projection
        to_delete = []
        for tid, trk in self._active_tracklets.items():
            if tid not in matched_tracks:
                trk.consecutive_invisible_frames += 1
                if trk.consecutive_invisible_frames > self.max_invisible:
                    trk.status = TrackStatus.TERMINATED
                    to_delete.append(tid)
                else:
                    trk.status = TrackStatus.OCCLUDED
                    # Velocity projection directly from last confirmed observation (non-recursive)
                    elapsed_t = max(0.0, timestamp_sec - trk.last_seen_timestamp)
                    vx, vy = trk.last_observed_velocity
                    # Gentle momentum damping across consecutive occluded frames
                    decay = max(0.2, 1.0 - (trk.consecutive_invisible_frames * 0.02))
                    dx = float(vx * elapsed_t * decay)
                    dy = float(vy * elapsed_t * decay)

                    ox1, oy1, ox2, oy2 = trk.last_observed_bbox
                    pred_x1 = max(-200.0, min(3000.0, ox1 + dx))
                    pred_y1 = max(-200.0, min(3000.0, oy1 + dy))
                    pred_x2 = max(pred_x1 + 10.0, min(3200.0, ox2 + dx))
                    pred_y2 = max(pred_y1 + 10.0, min(3200.0, oy2 + dy))
                    trk.current_bbox = (pred_x1, pred_y1, pred_x2, pred_y2)

                    if trk.last_observed_polygon:
                        trk.mask_polygon = [(float(px + dx), float(py + dy)) for px, py in trk.last_observed_polygon]
                    # Note: We do NOT append to trk.observed_trajectory during occlusion to prevent runaway velocity!

        for tid in to_delete:
            self._terminated_tracklets.append(self._active_tracklets.pop(tid))

        # Create new tracklets for unmatched detections
        for did in unmatched_dets:
            det = detections[did]
            local_id = f"{self.camera_id}_t{self._next_local_id:04d}"
            self._next_local_id += 1

            new_trk = CartonTracklet(
                camera_id=self.camera_id,
                local_track_id=local_id,
                first_seen_frame=frame_idx,
                first_seen_timestamp=timestamp_sec,
                last_seen_frame=frame_idx,
                last_seen_timestamp=timestamp_sec,
                current_bbox=det.bbox,
                current_confidence=det.confidence,
                status=TrackStatus.TENTATIVE if self.min_hits > 1 else TrackStatus.CONFIRMED,
                mask_polygon=det.mask_polygon,
                trajectory=deque([(det.center[0], det.center[1], timestamp_sec)], maxlen=60),
                total_visible_frames=1,
                consecutive_invisible_frames=0,
            )
            self._active_tracklets[local_id] = new_trk

        return list(self._active_tracklets.values())
