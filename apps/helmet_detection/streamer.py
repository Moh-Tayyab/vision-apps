"""Shared frame-buffer/streaming helpers for the apps.

Duplicated intentionally per app so each app stays fully independent
(senior requirement: no shared runtime coupling).
"""

from __future__ import annotations

import time
import threading
from typing import Generator, Optional

import cv2
import numpy as np


class FrameBuffer:
    """Thread-safe latest-frame store for pushed camera frames."""

    def __init__(self, max_frames: int = 10):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._max_frames = max_frames
        self._history: list[tuple[float, np.ndarray]] = []
        self._frame_count: int = 0

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()
            self._timestamp = time.time()
            self._frame_count += 1
            self._history.append((self._timestamp, frame.copy()))
            if len(self._history) > self._max_frames:
                self._history.pop(0)

    def get_latest(self) -> Optional[tuple[float, np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return None
            return self._timestamp, self._frame.copy()

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._frame is not None and (time.time() - self._timestamp < 5.0)


def apply_transform(frame: np.ndarray, mode: str | None) -> np.ndarray:
    """Apply an orientation correction to a frame.

    Modes: none, flip_h, flip_v, rotate_90_cw, rotate_90_ccw, rotate_180.
    """
    if not mode or mode == "none" or frame is None:
        return frame
    m = str(mode).lower().strip()
    if m in ("flip_h", "horizontal_flip", "mirror"):
        return cv2.flip(frame, 1)
    if m in ("flip_v", "vertical_flip"):
        return cv2.flip(frame, 0)
    if m in ("rotate_90_cw", "rotate_90", "cw", "90_cw", "90"):
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if m in ("rotate_90_ccw", "rotate_270", "ccw", "90_ccw", "270"):
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if m in ("rotate_180", "180"):
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def mjpeg_from_buffer(
    buffer: FrameBuffer,
    quality: int = 75,
    transform=None,
    fps_limit: float = 30.0,
) -> Generator[bytes, None, None]:
    """MJPEG multipart stream from a FrameBuffer; optional per-frame transform."""
    min_frame_time = 1.0 / max(1.0, fps_limit)
    last_sent_time = 0.0
    last_seen_ts = 0.0

    while True:
        result = buffer.get_latest()
        if result is None:
            time.sleep(0.04)
            continue

        ts, frame = result
        if ts == last_seen_ts:
            time.sleep(0.015)
            continue

        last_seen_ts = ts
        now = time.time()
        elapsed = now - last_sent_time
        if elapsed < min_frame_time:
            time.sleep(min_frame_time - elapsed)

        if transform is not None:
            frame = transform(frame)
            if frame is None:
                continue

        ok, out = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            continue

        last_sent_time = time.time()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + out.tobytes()
            + b"\r\n"
        )
