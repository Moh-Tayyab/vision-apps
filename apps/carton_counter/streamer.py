"""Video streaming: MJPEG and WebSocket with thread-safe frame buffer."""

from __future__ import annotations

import asyncio
import io
import time
import threading
from typing import AsyncGenerator, Generator, Optional

import cv2
import numpy as np


class FrameBuffer:
    """Thread-safe circular frame buffer."""

    def __init__(self, max_frames: int = 3):
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

    def get_history(self, count: int = 3) -> list[tuple[float, np.ndarray]]:
        with self._lock:
            return list(self._history[-count:])

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._frame is not None


class VideoSource:
    """OpenCV video source with auto-reconnect."""

    def __init__(self, source: str | int = 0, width: int = 1280, height: int = 720):
        self.source = source
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

    def open(self) -> bool:
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                return True
            try:
                src = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
                self._cap = cv2.VideoCapture(src)
                if not self._cap.isOpened():
                    return False
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return True
            except Exception:
                return False

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                if not self.open():
                    return None
            ret, frame = self._cap.read()
            if not ret:
                self.release()
                if not self.open():
                    return None
                ret, frame = self._cap.read()
                if not ret:
                    return None
            return frame

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def __del__(self):
        self.release()


class MobileCameraStream:
    """MJPEG streaming from video source for mobile camera access."""

    def __init__(self, source: str | int = 0, fps: int = 30):
        self.source = source
        self.fps = fps
        self._buffer = FrameBuffer(max_frames=10)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._video: Optional[VideoSource] = None

    def start(self) -> None:
        if self._running:
            return
        self._video = VideoSource(self.source)
        if not self._video.open():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._video is not None:
            self._video.release()
            self._video = None

    def _capture_loop(self) -> None:
        delay = 1.0 / self.fps
        while self._running and self._video is not None:
            frame = self._video.read()
            if frame is not None:
                self._buffer.update(frame)
            time.sleep(delay)

    def get_frame(self) -> Optional[np.ndarray]:
        result = self._buffer.get_latest()
        return result[1] if result else None

    def get_recent_frames(self, count: int = 3) -> list[np.ndarray]:
        history = self._buffer.get_history(count)
        return [f for _, f in history]

    def mjpeg_generator(self) -> Generator[bytes, None, None]:
        while self._running:
            result = self._buffer.get_latest()
            if result is None:
                time.sleep(0.05)
                continue
            _, frame = result
            _, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    @property
    def is_active(self) -> bool:
        return self._running and self._buffer.is_active

    @property
    def frame_count(self) -> int:
        return self._buffer.frame_count


class MJPEGServer:
    """Standalone MJPEG HTTP server for streaming."""

    def __init__(self, stream: MobileCameraStream, host: str = "0.0.0.0", port: int = 8081):
        self.stream = stream
        self.host = host
        self.port = port
        self._server: Optional[object] = None

    async def stream_response(self) -> AsyncGenerator[bytes, None]:
        loop = asyncio.get_event_loop()
        while True:
            result = await loop.run_in_executor(None, self.stream.get_frame)
            if result is None:
                await asyncio.sleep(0.05)
                continue
            _, buffer = cv2.imencode(
                ".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )


def mjpeg_from_buffer(
    buffer: FrameBuffer,
    quality: int = 80,
    transform=None,
    fps_limit: float = 20.0,
) -> Generator[bytes, None, None]:
    """Yield an MJPEG multipart stream from a FrameBuffer.

    ``transform(frame) -> frame`` allows annotated output (e.g. detection boxes).
    Frames are cached by timestamp to avoid redundant inferences on static buffers.
    """
    last_ts = 0.0
    last_output_bytes: Optional[bytes] = None
    min_interval = 1.0 / fps_limit

    while True:
        result = buffer.get_latest()
        if result is None:
            time.sleep(0.05)
            continue
        ts, frame = result
        if ts != last_ts or last_output_bytes is None:
            processed = frame
            if transform is not None:
                try:
                    transformed = transform(frame)
                    if transformed is not None:
                        processed = transformed
                except Exception:
                    processed = frame
            ok, out = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                last_output_bytes = out.tobytes()
            last_ts = ts

        if last_output_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + last_output_bytes
                + b"\r\n"
            )
        time.sleep(min_interval)


async def websocket_stream(
    websocket: "WebSocket",
    stream: MobileCameraStream,
) -> None:
    """Stream frames over WebSocket as base64 JPEG."""
    import base64
    from starlette.websockets import WebSocket

    await websocket.accept()
    loop = asyncio.get_event_loop()
    try:
        while True:
            result = await loop.run_in_executor(None, stream.get_frame)
            if result is None:
                await asyncio.sleep(0.05)
                continue
            _, frame = result
            _, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
            await websocket.send_json({"type": "frame", "data": b64})
    except Exception:
        pass
    finally:
        await websocket.close()
