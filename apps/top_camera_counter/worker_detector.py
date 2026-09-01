"""Worker hand and pose detection for pick event logic.

Uses MediaPipe for hand tracking and pose estimation.
Detects when a worker is reaching for or picking a carton.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


@dataclass
class HandDetection:
    """Detected hand with position and confidence."""
    hand_id: int
    position: Tuple[float, float]  # x, y center
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    is_left: bool


class WorkerDetector:
    """Detects worker hands and pose for pick event detection."""

    def __init__(self, use_hands: bool = True):
        self.use_hands = use_hands and HAS_MEDIAPIPE

        if not HAS_MEDIAPIPE:
            print("Warning: MediaPipe not available, worker detection disabled")

        if self.use_hands:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        self.hand_history: List[HandDetection] = []
        self.max_history = 30

    def _mediapipe_to_px(
        self, landmark, w: int, h: int
    ) -> Tuple[float, float]:
        return (landmark.x * w, landmark.y * h)

    def detect_hands(self, frame: np.ndarray) -> List[HandDetection]:
        """Detect hands in frame."""
        if not self.use_hands:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hands = []
        if results.multi_hand_landmarks:
            for hand_idx, hand_lms in enumerate(results.multi_hand_landmarks):
                h, w = frame.shape[:2]
                positions = [
                    self._mediapipe_to_px(lm, w, h) for lm in hand_lms.landmark
                ]
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                handedness = results.multi_handedness[hand_idx]
                is_left = handedness.classification[0].label == "Left"
                confidence = handedness.classification[0].score

                hands.append(HandDetection(
                    hand_id=hand_idx,
                    position=center,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    is_left=is_left
                ))

        return hands

    def detect(self, frame: np.ndarray) -> dict:
        """Run all worker detections."""
        hands = self.detect_hands(frame)

        self.hand_history.extend(hands)
        if len(self.hand_history) > self.max_history:
            self.hand_history = self.hand_history[-self.max_history:]

        return {
            "hands": hands,
            "hand_reaching": self._is_reaching_towards_pallet(hands),
        }

    def _is_reaching_towards_pallet(
        self, hands: List[HandDetection]
    ) -> bool:
        """Check if any hand is in the pallet ROI area."""
        # This will be configured by the live_counter with actual ROI
        return False

    def get_hand_velocity(self) -> Optional[Tuple[float, float]]:
        """Calculate velocity of most recent hand movement."""
        if len(self.hand_history) < 2:
            return None

        prev = self.hand_history[-2]
        curr = self.hand_history[-1]
        dx = curr.position[0] - prev.position[0]
        dy = curr.position[1] - prev.position[1]
        return (dx, dy)

    def reset(self):
        """Reset detection state."""
        self.hand_history.clear()
