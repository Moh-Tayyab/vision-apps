from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import time


class TripwireCounter:
    """
    Directional tripwire counter for tracking moving objects across a vertical line.
    
    Directions:
      - Left-to-Right (x_old < line_x and x_new >= line_x): +1 (e.g. Loaded into truck)
      - Right-to-Left (x_old > line_x and x_new <= line_x): -1 (e.g. Returned from truck)
    """

    def __init__(
        self,
        line_x: int,
        hysteresis: int = 15,
        history_len: int = 30,
        cooldown_frames: int = 15,
    ):
        """
        Args:
            line_x: X-coordinate of the vertical virtual line.
            hysteresis: Pixel buffer around the line to prevent noise oscillation.
            history_len: Max past positions to maintain per track ID.
            cooldown_frames: Number of frames before the same track ID can trigger another crossing.
        """
        self.line_x = line_x
        self.hysteresis = hysteresis
        self.history_len = history_len
        self.cooldown_frames = cooldown_frames

        # State tracking
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.history_len))
        self.track_side: Dict[int, str] = {}  # 'left' or 'right'
        self.last_cross_frame: Dict[int, int] = {}
        self.track_classes: Dict[int, str] = {}

        # Counters
        self.total_in = 0    # Left -> Right (+1)
        self.total_out = 0   # Right -> Left (-1)
        self.events: List[dict] = []  # Log of crossing events

        # Active toast notification for UI
        self.recent_event: Optional[dict] = None
        self.recent_event_expiry: float = 0.0

    @property
    def net_count(self) -> int:
        return self.total_in - self.total_out

    def set_line_x(self, new_x: int) -> None:
        """Update virtual line X coordinate interactively."""
        self.line_x = new_x

    def reset_counts(self) -> None:
        """Reset all counters and history."""
        self.total_in = 0
        self.total_out = 0
        self.track_history.clear()
        self.track_side.clear()
        self.last_cross_frame.clear()
        self.events.clear()
        self.recent_event = None

    def update(
        self,
        tracked_objects: List[Tuple[int, Tuple[float, float, float, float], str, float]],
        frame_idx: int,
    ) -> List[dict]:
        """
        Update tracker states with detections in the current frame and detect line crossings.

        Args:
            tracked_objects: List of tuples (track_id, (x1, y1, x2, y2), class_name, confidence)
            frame_idx: Current video frame index.

        Returns:
            List of crossing event dicts triggered in this frame.
        """
        current_frame_events = []
        active_ids = set()

        for track_id, (x1, y1, x2, y2), cls_name, conf in tracked_objects:
            active_ids.add(track_id)
            self.track_classes[track_id] = cls_name
            
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            history = self.track_history[track_id]
            history.append((center_x, center_y))

            # Initial side assignment
            if track_id not in self.track_side:
                if center_x < self.line_x - self.hysteresis:
                    self.track_side[track_id] = "left"
                elif center_x > self.line_x + self.hysteresis:
                    self.track_side[track_id] = "right"
                continue

            last_side = self.track_side[track_id]
            last_frame = self.last_cross_frame.get(track_id, -self.cooldown_frames)

            # Check for cooldown
            if frame_idx - last_frame < self.cooldown_frames:
                continue

            # Check Left -> Right Crossing (+1 Loaded)
            if last_side == "left" and center_x >= self.line_x + self.hysteresis:
                self.total_in += 1
                self.track_side[track_id] = "right"
                self.last_cross_frame[track_id] = frame_idx
                
                event = {
                    "frame": frame_idx,
                    "timestamp": time.time(),
                    "track_id": track_id,
                    "class_name": cls_name,
                    "direction": "IN",
                    "delta": +1,
                    "position": (center_x, center_y),
                    "total_in": self.total_in,
                    "total_out": self.total_out,
                    "net_count": self.net_count,
                }
                self.events.append(event)
                current_frame_events.append(event)
                self.recent_event = event
                self.recent_event_expiry = time.time() + 1.8

            # Check Right -> Left Crossing (-1 Returned)
            elif last_side == "right" and center_x <= self.line_x - self.hysteresis:
                self.total_out += 1
                self.track_side[track_id] = "left"
                self.last_cross_frame[track_id] = frame_idx
                
                event = {
                    "frame": frame_idx,
                    "timestamp": time.time(),
                    "track_id": track_id,
                    "class_name": cls_name,
                    "direction": "OUT",
                    "delta": -1,
                    "position": (center_x, center_y),
                    "total_in": self.total_in,
                    "total_out": self.total_out,
                    "net_count": self.net_count,
                }
                self.events.append(event)
                current_frame_events.append(event)
                self.recent_event = event
                self.recent_event_expiry = time.time() + 1.8

        return current_frame_events
