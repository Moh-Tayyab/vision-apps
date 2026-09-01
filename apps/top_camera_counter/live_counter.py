"""Live carton counting system.

Integrates YOLO detection, ByteTrack tracking, worker detection,
and state machine for real-time pallet carton counting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detector import CartonDetector, DetectorError
from tracker import ByteTracker, Track
from worker_detector import WorkerDetector
from state_machine import CartonStateMachine, CartonState


@dataclass
class PalletROI:
    """Region of Interest for the pallet area."""
    x1: int
    y1: int
    x2: int
    y2: int
    layer_heights: List[float] = field(default_factory=list)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class LiveCountResult:
    """Result from live counting."""
    total_active: int
    total_picked: int
    layer_counts: Dict[int, int]
    cartons_by_state: Dict[str, int]
    hand_detected: bool
    picking_in_progress: bool
    frame_time_ms: float
    tracks: List[dict]
    events: List[dict]


class LiveCartonCounter:
    """Real-time carton counter with pick detection."""

    def __init__(
        self,
        detector: CartonDetector,
        pallet_roi: Optional[PalletROI] = None,
        tracker_max_age: int = 30,
        tracker_min_hits: int = 3,
        occlusion_timeout: int = 30,
        hand_proximity_threshold: float = 100.0,
    ):
        self.detector = detector
        self.pallet_roi = pallet_roi
        self.tracker = ByteTracker(
            max_age=tracker_max_age,
            min_hits=tracker_min_hits,
        )
        self.worker_detector = WorkerDetector()
        self.state_machine = CartonStateMachine(
            occlusion_timeout=occlusion_timeout,
            hand_proximity_threshold=hand_proximity_threshold,
        )
        self.frame_count = 0
        self.initial_cartons: Optional[int] = None

    def set_initial_count(self, count: int):
        """Set initial carton count for the pallet."""
        self.initial_cartons = count

    def process_frame(self, frame: np.ndarray) -> LiveCountResult:
        """Process a single frame and return count result."""
        start = time.perf_counter()

        # Detect cartons
        try:
            boxes, inference_ms = self.detector.detect(frame)
        except DetectorError:
            boxes = []
            inference_ms = 0.0

        # Convert boxes to detection format
        detections = [(b["x1"], b["y1"], b["x2"], b["y2"], b["confidence"]) for b in boxes]

        # Filter to ROI if defined
        if self.pallet_roi:
            detections = [
                d for d in detections
                if self.pallet_roi.contains(
                    (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
                )
            ]

        # Update tracker
        active_tracks = self.tracker.update(detections)

        # Detect worker hands/pose
        worker_result = self.worker_detector.detect(frame)

        # Extract hand positions
        hand_positions = [h.position for h in worker_result["hands"]]
        hand_velocity = self.worker_detector.get_hand_velocity()

        # Update state machine
        detected_ids = [t.track_id for t in active_tracks]
        detected_bboxes = {t.track_id: t.bbox for t in active_tracks}
        detected_confidences = {t.track_id: t.confidence for t in active_tracks}

        events = self.state_machine.update(
            detected_ids=detected_ids,
            detected_bboxes=detected_bboxes,
            detected_confidences=detected_confidences,
            hand_positions=hand_positions,
            hand_velocity=hand_velocity,
        )

        # Build result
        active_cartons = self.state_machine.get_active_cartons()
        layer_counts = self.state_machine.get_layer_counts()

        cartons_by_state = {}
        for state in CartonState:
            count = len(self.state_machine.get_cartons_by_state(state))
            if count > 0:
                cartons_by_state[state.value] = count

        frame_time = (time.perf_counter() - start) * 1000.0

        return LiveCountResult(
            total_active=len(active_cartons),
            total_picked=self.state_machine.picked_count,
            layer_counts=layer_counts,
            cartons_by_state=cartons_by_state,
            hand_detected=len(hand_positions) > 0,
            picking_in_progress=any(
                t.state == CartonState.BEING_PICKED
                for t in self.state_machine.tracks.values()
            ),
            frame_time_ms=frame_time,
            tracks=[
                {
                    "id": t.track_id,
                    "bbox": t.bbox,
                    "state": t.state.value,
                    "row": t.row,
                    "layer": t.layer,
                }
                for t in active_cartons
            ],
            events=events,
        )

    def annotate_frame(
        self, frame: np.ndarray, result: LiveCountResult
    ) -> np.ndarray:
        """Draw annotations on frame."""
        vis = frame.copy()

        # Draw pallet ROI
        if self.pallet_roi:
            cv2.rectangle(
                vis,
                (self.pallet_roi.x1, self.pallet_roi.y1),
                (self.pallet_roi.x2, self.pallet_roi.y2),
                (255, 255, 0), 2
            )

        # Draw tracks
        for track in result.tracks:
            x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
            state = track["state"]
            color = {
                "present": (0, 255, 0),
                "being_picked": (0, 165, 255),
                "occluded": (128, 128, 128),
                "removed": (0, 0, 255),
            }.get(state, (255, 255, 255))

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track['id']} {state[:8]}"
            cv2.putText(
                vis, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Draw hand positions
        for hand in self.worker_detector.hand_history[-5:]:
            hx, hy = int(hand.position[0]), int(hand.position[1])
            cv2.circle(vis, (hx, hy), 15, (255, 0, 255), 3)

        # Draw stats
        y_offset = 30
        stats = [
            f"Active: {result.total_active}",
            f"Picked: {result.total_picked}",
            f"Hand: {'YES' if result.hand_detected else 'NO'}",
            f"Picking: {'YES' if result.picking_in_progress else 'NO'}",
            f"Time: {result.frame_time_ms:.0f}ms",
        ]
        for stat in stats:
            cv2.putText(
                vis, stat, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            y_offset += 30

        return vis

    def reset(self):
        """Reset all counters and state."""
        self.tracker.reset()
        self.worker_detector.reset()
        self.state_machine.reset()
        self.frame_count = 0
        self.initial_cartons = None
