"""Object tracking for live carton counting.

Uses ByteTrack-inspired algorithm for robust multi-object tracking with unique IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Track:
    """A tracked object with unique ID and history."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class ByteTracker:
    """Simplified ByteTrack-inspired tracker.

    Assigns unique IDs to detected objects and maintains tracks across frames.
    Uses IoU-based matching for simplicity.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def _compute_iou(
        self, box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float]
    ) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def update(
        self, detections: List[Tuple[float, float, float, float, float]]
    ) -> List[Track]:
        """Update tracker with new detections.

        Args:
            detections: List of (x1, y1, x2, y2, confidence) tuples.

        Returns:
            List of active tracks (with enough hits).
        """
        # Match detections to existing tracks using IoU
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        for det_idx in range(len(detections)):
            best_iou = 0.0
            best_track_id = None
            for track_id in unmatched_tracks:
                iou = self._compute_iou(
                    detections[det_idx][:4],
                    self.tracks[track_id].bbox
                )
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            if best_iou >= self.iou_threshold and best_track_id is not None:
                matched.append((det_idx, best_track_id))
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(best_track_id)

        # Update matched tracks
        for det_idx, track_id in matched:
            track = self.tracks[track_id]
            track.bbox = detections[det_idx][:4]
            track.confidence = detections[det_idx][4]
            track.hits += 1
            track.time_since_update = 0
            track.history.append(track.bbox)
            if len(track.history) > 50:
                track.history.pop(0)

        # Update unmatched tracks (age them)
        for track_id in unmatched_tracks:
            self.tracks[track_id].age += 1
            self.tracks[track_id].time_since_update += 1

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = Track(
                track_id=track_id,
                bbox=detections[det_idx][:4],
                confidence=detections[det_idx][4],
                history=[detections[det_idx][:4]]
            )

        # Remove dead tracks
        dead_ids = [
            tid for tid, t in self.tracks.items()
            if t.time_since_update > self.max_age
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        # Return active tracks with enough hits
        return [
            t for t in self.tracks.values()
            if t.hits >= self.min_hits
        ]

    def get_all_tracks(self) -> List[Track]:
        """Get all current tracks (regardless of hit count)."""
        return list(self.tracks.values())

    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
