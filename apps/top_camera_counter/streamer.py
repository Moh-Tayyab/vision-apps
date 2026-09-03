"""Frame buffer and image transform utilities."""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


class FrameBuffer:
    """Thread-safe frame buffer for latest frame."""

    def __init__(self, max_frames: int = 2):
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._timestamp: float = 0.0
        self.frame_count: int = 0

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()
            self._timestamp = time.time()
            self.frame_count += 1

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def timestamp(self) -> float:
        return self._timestamp


def apply_transform(frame: np.ndarray, mode: str) -> np.ndarray:
    """Apply orientation transform to frame."""
    if mode == "none" or not mode:
        return frame
    elif mode == "rotate_90":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif mode == "rotate_180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif mode == "rotate_270":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif mode == "flip_h":
        return cv2.flip(frame, 1)
    elif mode == "flip_v":
        return cv2.flip(frame, 0)
    elif mode == "flip_both":
        return cv2.flip(frame, -1)
    return frame
