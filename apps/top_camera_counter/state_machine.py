"""State machine for carton lifecycle management.

Tracks each carton through states: PRESENT -> BEING_PICKED -> REMOVED
Handles occlusion and temporary disappearances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time


class CartonState(Enum):
    """Carton states in the picking lifecycle."""
    PRESENT = "present"
    BEING_PICKED = "being_picked"
    REMOVED = "removed"
    OCCLUDED = "occluded"


@dataclass
class CartonTrack:
    """Tracks a single carton through its lifecycle."""
    track_id: int
    state: CartonState = CartonState.PRESENT
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    confidence: float = 0.0
    row: int = 0
    layer: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_state_change: float = field(default_factory=time.time)
    occluded_frames: int = 0
    picked_by_hand: bool = False
    hand_nearby_frames: int = 0


class CartonStateMachine:
    """Manages state transitions for all tracked cartons.

    State transitions:
        PRESENT -> BEING_PICKED: When hand is detected near carton
        BEING_PICKED -> REMOVED: When carton disappears AND hand moving away
        PRESENT -> OCCLUDED: When carton temporarily disappears (no hand)
        OCCLUDED -> PRESENT: When carton reappears
        OCCLUDED -> REMOVED: When disappeared too long
    """

    def __init__(
        self,
        occlusion_timeout: int = 8,
        removal_timeout: int = 8,
        hand_proximity_threshold: float = 100.0,
    ):
        self.tracks: Dict[int, CartonTrack] = {}
        self.occlusion_timeout = occlusion_timeout
        self.removal_timeout = removal_timeout
        self.hand_proximity_threshold = hand_proximity_threshold
        self.picked_count = 0
        self.pick_events: List[dict] = []

    def _distance(
        self, box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float]
    ) -> float:
        """Distance between box centers."""
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2
        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    def update(
        self,
        detected_ids: List[int],
        detected_bboxes: Dict[int, Tuple[float, float, float, float]],
        detected_confidences: Dict[int, float],
        hand_positions: List[Tuple[float, float]],
        hand_velocity: Optional[Tuple[float, float]] = None,
    ) -> List[dict]:
        """Update all carton states based on detections and hand info.

        Returns list of pick events detected in this frame.
        """
        current_time = time.time()
        events = []

        # Update existing tracks
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]

            if track_id in detected_ids:
                # Carton is visible
                track.bbox = detected_bboxes[track_id]
                track.confidence = detected_confidences[track_id]
                track.last_seen = current_time
                track.occluded_frames = 0

                # Check if hand is near this carton
                hand_near = any(
                    self._distance(track.bbox, (hx, hy, hx, hy))
                    < self.hand_proximity_threshold
                    for hx, hy in hand_positions
                )

                if hand_near:
                    track.hand_nearby_frames += 1
                    if track.state == CartonState.PRESENT:
                        track.state = CartonState.BEING_PICKED
                        track.picked_by_hand = True
                        track.last_state_change = current_time
                else:
                    track.hand_nearby_frames = max(0, track.hand_nearby_frames - 1)

                # State transitions
                if track.state == CartonState.BEING_PICKED:
                    if not hand_near and track.hand_nearby_frames == 0:
                        # Hand moved away - carton was picked
                        track.state = CartonState.REMOVED
                        track.last_state_change = current_time
                        self.picked_count += 1
                        event = {
                            "track_id": track_id,
                            "event": "picked",
                            "time": current_time,
                            "row": track.row,
                            "layer": track.layer,
                        }
                        events.append(event)
                        self.pick_events.append(event)

                elif track.state == CartonState.OCCLUDED:
                    # Carton reappeared
                    track.state = CartonState.PRESENT

            else:
                # Carton not detected
                track.occluded_frames += 1

                if track.state == CartonState.PRESENT:
                    # Check if hand is nearby (might be picking)
                    hand_near = any(
                        self._distance(track.bbox, (hx, hy, hx, hy))
                        < self.hand_proximity_threshold
                        for hx, hy in hand_positions
                    )
                    if hand_near:
                        track.state = CartonState.BEING_PICKED
                        track.picked_by_hand = True
                        track.last_state_change = current_time
                    elif track.occluded_frames > self.occlusion_timeout:
                        track.state = CartonState.REMOVED
                        track.last_state_change = current_time
                        self.picked_count += 1
                        events.append({
                            "track_id": track_id,
                            "event": "removed_timeout",
                            "time": current_time,
                            "row": track.row,
                            "layer": track.layer,
                        })
                    else:
                        track.state = CartonState.OCCLUDED

                elif track.state == CartonState.BEING_PICKED:
                    if hand_velocity:
                        # Hand is moving away - likely picked
                        speed = (hand_velocity[0] ** 2 + hand_velocity[1] ** 2) ** 0.5
                        if speed > 5.0:
                            track.state = CartonState.REMOVED
                            track.last_state_change = current_time
                            self.picked_count += 1
                            events.append({
                                "track_id": track_id,
                                "event": "picked_velocity",
                                "time": current_time,
                                "row": track.row,
                                "layer": track.layer,
                            })
                    elif track.occluded_frames > self.removal_timeout:
                        track.state = CartonState.REMOVED
                        track.last_state_change = current_time

                elif track.state == CartonState.OCCLUDED:
                    if track.occluded_frames > self.occlusion_timeout:
                        track.state = CartonState.REMOVED
                        events.append({
                            "track_id": track_id,
                            "event": "removed_occluded",
                            "time": current_time,
                            "row": track.row,
                            "layer": track.layer,
                        })

        # Add new tracks
        for track_id in detected_ids:
            if track_id not in self.tracks:
                self.tracks[track_id] = CartonTrack(
                    track_id=track_id,
                    bbox=detected_bboxes[track_id],
                    confidence=detected_confidences[track_id],
                )

        # Cleanup removed tracks that are older than 5 seconds
        for tid in list(self.tracks.keys()):
            t = self.tracks[tid]
            if t.state == CartonState.REMOVED and (current_time - t.last_state_change > 5.0):
                del self.tracks[tid]

        return events

    def get_active_cartons(self, include_occluded: bool = False) -> List[CartonTrack]:
        """Get all cartons actively present or being picked (excludes occluded unless requested)."""
        if include_occluded:
            return [t for t in self.tracks.values() if t.state != CartonState.REMOVED]
        return [
            t for t in self.tracks.values()
            if t.state in (CartonState.PRESENT, CartonState.BEING_PICKED)
        ]

    def get_cartons_by_state(self, state: CartonState) -> List[CartonTrack]:
        """Get cartons in a specific state."""
        return [t for t in self.tracks.values() if t.state == state]

    def get_layer_counts(self) -> Dict[int, int]:
        """Count active cartons per layer."""
        counts = {}
        for track in self.get_active_cartons(include_occluded=False):
            counts[track.layer] = counts.get(track.layer, 0) + 1
        return counts

    def reset(self):
        """Reset all state."""
        self.tracks.clear()
        self.picked_count = 0
        self.pick_events.clear()
