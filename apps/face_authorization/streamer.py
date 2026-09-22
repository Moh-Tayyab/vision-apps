"""Shared frame-buffer/streaming helpers for Face Authorization.

Duplicated intentionally per app so each app stays fully independent
(senior requirement: no shared runtime coupling).
"""

from __future__ import annotations

import os
import time
import threading
from typing import Generator, Optional

import cv2
import numpy as np


class FrameBuffer:
    """Thread-safe latest-frame store for pushed camera frames."""

    def __init__(self, max_frames: int = 3):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._max_frames = max_frames
        self._frame_count: int = 0

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._timestamp = time.time()
            self._frame_count += 1

    def get_latest(self) -> Optional[tuple[float, np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return None
            return self._timestamp, self._frame

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
    Used to fix mirrored / rotated mobile IP-webcam feeds (e.g. when the
    view appears shifted to the left or flipped relative to the real scene).
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


class MobileCameraStream:
    """MJPEG / device streaming engine for mobile IP webcam (matches carton_counter App 1).

    Pulls frames directly from a video source (IP webcam URL, USB device, RTSP) into a
    shared FrameBuffer. This is the unified live-stream method shared by all 3 apps.
    """

    def __init__(self, source: str | int = 0, fps: int = 30, transform: str | None = None):
        self.source = source
        self.fps = fps
        self._transform = transform
        self._buffer = FrameBuffer(max_frames=10)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._video: Optional["_VideoSource"] = None

    def start(self) -> None:
        if self._running:
            return
        self._video = _VideoSource(self.source)
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
        delay = 1.0 / max(1, self.fps)
        # Downscale right after read: dramatically cuts JPEG decode/resize/encode
        # and inference cost when sources stream large frames (e.g. phone 1440x1440).
        max_capture_width = int(os.getenv("CAPTURE_MAX_WIDTH", "960"))
        while self._running and self._video is not None:
            frame = self._video.read()
            if frame is not None:
                h, w = frame.shape[:2]
                if w > max_capture_width:
                    scale = max_capture_width / w
                    frame = cv2.resize(frame, (max_capture_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
                if self._transform:
                    frame = apply_transform(frame, self._transform)
                self._buffer.update(frame)
            time.sleep(delay)

    def get_frame(self) -> Optional[np.ndarray]:
        result = self._buffer.get_latest()
        return result[1] if result else None

    def mjpeg_generator(self) -> Generator[bytes, None, None]:
        while self._running:
            result = self._buffer.get_latest()
            if result is None:
                time.sleep(0.05)
                continue
            _, frame = result
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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

    @property
    def buffer(self) -> FrameBuffer:
        return self._buffer


class _VideoSource:
    """Video source with auto-reconnect.

    HTTP(S) URLs use a dedicated MJPEG reader thread that never blocks the caller:
    network hiccups are absorbed, and the feed automatically reconnects (fixed the
    cv2 read() stall that froze stream stop() and left the stream permanently wedged).
    Local devices (0/1/2, /dev/video*) fall back to OpenCV capture.
    """

    def __init__(self, source: str | int = 0, width: int = 1280, height: int = 720):
        self.source = source
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._http = _MjpegSource(self.source)
        self._lock = threading.Lock()
        self._is_http = isinstance(source, str) and source.startswith(("http://", "https://"))

    def open(self) -> bool:
        if self._is_http:
            return self._http.open()
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                return True
            try:
                src = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
                self._cap = cv2.VideoCapture(src)
                if not self._cap.isOpened():
                    return False
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return True
            except Exception:
                return False

    def read(self) -> Optional[np.ndarray]:
        if self._is_http:
            return self._http.read()
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
        self._http.close()
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None


class _MjpegSource:
    """Minimal HTTP MJPEG reader that decodes JPEG boundaries over an HTTP stream.

    A background thread owns the TCP socket and does the blocking reads; callers just
    grab the latest cached frame, so a dead/stalled phone feed can neither wedge read()
    nor stop(). Reconnects automatically with a short backoff.
    """

    _SOI = b"\xff\xd8"
    _EOI = b"\xff\xd9"

    def __init__(self, url: str, reconnect_delay: float = 1.0):
        self.url = url
        self._reconnect_delay = reconnect_delay
        self._latest: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def open(self) -> bool:
        if self._running:
            return True
        if self._test_open():
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return True
        return False

    def read(self) -> Optional[np.ndarray]:
        result = self.get_latest()
        return result[1] if result else None

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # -- internal ----------------------------------------------------------

    def _test_open(self) -> bool:
        try:
            import urllib.request

            with urllib.request.urlopen(self.url, timeout=5) as _:
                return True
        except Exception:
            return False

    def _loop(self) -> None:
        while self._running:
            try:
                self._stream_once()
            except Exception:
                pass
            if not self._running:
                break
            time.sleep(self._reconnect_delay)

    def _stream_once(self) -> None:
        import re
        import urllib.request

        req = urllib.request.Request(self.url, headers={"Connection": "keep-alive"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            ctype = resp.headers.get("Content-Type", "")
            bm = re.search(r"boundary[ \t]*=[ \t]*[\"']?([^;\"'\r\n]+)", ctype)
            buf = b""
            if bm:
                boundary = b"--" + bm.group(1).encode()
            else:
                boundary = b""
            while self._running:
                chunk = resp.read(65536)
                if not chunk:
                    break
                buf += chunk
                if not boundary:
                    self._frag_scan(buf)
                    buf = buf[-4096:]
                    continue
                if boundary not in buf:
                    if len(buf) > max(65536, len(boundary) * 4):
                        buf = buf[-len(boundary):]
                    continue
                # each boundary terminates a frame payload: "--boundary\r\n<hdrs>\r\n<bytes>\r\n--boundary"
                idx = 0
                while True:
                    s = buf.find(boundary, idx)
                    if s == -1:
                        break
                    body = buf[idx:s]
                    if body:
                        data = self._extract_jpeg(body)
                        if data is not None:
                            frame = self._decode(data)
                            if frame is not None:
                                with self._lock:
                                    self._latest = frame
                                    self._latest_ts = time.time()
                    idx = s + len(boundary)
                buf = buf[idx:]

    def _extract_jpeg(self, body: bytes) -> Optional[bytes]:
        hdr = body.find(b"\r\n\r\n")
        if hdr != -1:
            body = body[hdr + 4:]
        s = body.find(self._SOI)
        e = body.rfind(self._EOI)
        if s != -1 and e > s:
            return body[s : e + 2]
        return body if s != -1 else None

    def _frag_scan(self, buf: bytes) -> None:
        """Fallback SOI/EOI scanner for feeds without a multipart boundary."""
        idx = 0
        gathering = False
        for p in range(len(buf) - 1):
            if not gathering and buf[p : p + 2] == self._SOI:
                gathering = True
                idx = p
            elif gathering and buf[p : p + 2] == self._EOI:
                frame = self._decode(buf[idx : p + 2])
                if frame is not None:
                    with self._lock:
                        self._latest = frame
                        self._latest_ts = time.time()
                gathering = False

    def _decode(self, jpeg: bytes) -> Optional[np.ndarray]:
        try:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def get_latest(self) -> Optional[tuple]:
        with self._lock:
            if self._latest is None:
                return None
            return (self._latest_ts, self._latest.copy())


def _generate_standby_frame(w: int = 640, h: int = 480) -> np.ndarray:
    """Generate a clean dark placeholder canvas when no camera frames are arriving."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Dark slate background
    img[:] = (30, 24, 15)

    # Grid / aesthetic frame border
    cv2.rectangle(img, (15, 15), (w - 15, h - 15), (60, 50, 40), 2)
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), (45, 38, 28), 1)

    # Text headers
    title = "FACE AUTHORIZATION FEED"
    subtitle = "STANDBY - WAITING FOR CAMERA STREAM"
    tip1 = "Option 1: In Live Detection tab, choose Direct USB and Connect"
    tip2 = "Option 2: Open /mobile in phone browser to stream camera"

    cv2.putText(img, title, (w // 2 - 180, h // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
    cv2.putText(img, subtitle, (w // 2 - 210, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (251, 191, 36), 1)
    
    cv2.line(img, (80, h // 2 + 15), (w - 80, h // 2 + 15), (70, 60, 50), 1)
    cv2.putText(img, tip1, (60, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)
    cv2.putText(img, tip2, (60, h // 2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)

    # Pulsing timestamp / dot to show stream is live
    ts_str = time.strftime("%H:%M:%S")
    cv2.putText(img, f"LIVE ENGINE: {ts_str}", (w - 180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (34, 197, 94), 1)
    cv2.circle(img, (w - 195, 36), 4, (34, 197, 94), -1)

    return img


_ENCODE_CACHE_LOCK = threading.Lock()
_ENCODE_CACHE: dict = {}


def _encoded_jpeg(ts: float, frame, quality: int) -> bytes:
    """Encode once per unique frame timestamp; all clients share the same JPEG bytes.

    Prevents N connected browsers from each re-encoding the same frame (the main
    cause of sluggish MJPEG streams with multiple viewers).
    """
    with _ENCODE_CACHE_LOCK:
        if _ENCODE_CACHE.get("ts") == ts:
            return _ENCODE_CACHE["jpeg"]
    ok, out = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return b""
    with _ENCODE_CACHE_LOCK:
        _ENCODE_CACHE["ts"] = ts
        _ENCODE_CACHE["jpeg"] = out.tobytes()
    return _ENCODE_CACHE["jpeg"]


def mjpeg_from_buffer(
    buffer: FrameBuffer,
    quality: int = 65,
    transform=None,
    max_width: int = 854,
) -> Generator[bytes, None, None]:
    """MJPEG multipart stream from a FrameBuffer; optimized for real-time low-latency playback.

    Encodes each unique frame once (shared across all viewers) and downscales to
    ``max_width`` before JPEG encoding to drastically reduce CPU load.
    """
    last_ts_seen = -1.0
    while True:
        result = buffer.get_latest()
        is_fresh = False
        if result is not None:
            ts, frame = result
            if time.time() - ts < 5.0:
                is_fresh = True

        if not is_fresh:
            frame = _generate_standby_frame()
            last_ts_seen = -1.0
            time.sleep(0.08)
        else:
            # Skip identical frames: only encode when a new frame has arrived.
            if ts == last_ts_seen:
                time.sleep(0.004)
                continue
            last_ts_seen = ts
            if transform is not None:
                frame = transform(frame)
                if frame is None:
                    time.sleep(0.01)
                    continue
            # Downscale large frames before encoding (biggest CPU saver).
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

        # Shared encode: first client encodes, rest reuse the same JPEG bytes.
        payload = _encoded_jpeg(last_ts_seen, frame, quality)
        if not payload:
            time.sleep(0.01)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + payload
            + b"\r\n"
        )
