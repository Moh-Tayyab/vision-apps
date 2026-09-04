"""Modular Camera Source Manager for Face Authorization (App 3).

Supports:
- USB / V4L2 Webcams (/dev/video0, 0, 1, 2)
- Mobile IP Webcam via USB Tethering or Wi-Fi (http://ip:8080/video or DroidCam)
- RTSP Streams (rtsp://user:pass@ip:port/h264)
- Local Video Files (looping video for testing)
- Browser-based Mobile Ingest (/mobile & /ingest/frame)

Features:
- Independent background capture workers with thread safety
- Exponential backoff auto-reconnect on stream drop (e.g. USB cable pulled)
- Real-time health monitoring: FPS, last frame timestamp, connection status
- USB device auto-discovery on Linux (/dev/video*)
"""

from __future__ import annotations

import glob
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np

from streamer import FrameBuffer, apply_transform


# Orientation correction applied to every captured frame (set via /camera/transform).
_ORIENTATION_TRANSFORM: str = "none"


def set_orientation_transform(mode: str) -> None:
    global _ORIENTATION_TRANSFORM
    _ORIENTATION_TRANSFORM = (mode or "none").lower()


def _apply_orientation(frame: "np.ndarray") -> "np.ndarray":
    return apply_transform(frame, _ORIENTATION_TRANSFORM)


def _normalize_uri(source_type: str, uri: Union[str, int]) -> str:
    """Normalize input URI for HTTP MJPEG, RTSP, USB, or Video sources."""
    if source_type == "usb":
        return str(uri)
    if source_type in ("mobile", "video_file"):
        return str(uri)

    uri_str = str(uri).strip()
    if not uri_str:
        return ""

    if not uri_str.startswith(("http://", "https://", "rtsp://")):
        if source_type == "rtsp":
            uri_str = f"rtsp://{uri_str}"
        else:
            uri_str = f"http://{uri_str}"

    if source_type == "http_mjpeg":
        parsed = urlparse(uri_str)
        if not parsed.path or parsed.path == "/":
            if parsed.port in (8080, 4747, 8081, 8000) or not parsed.path:
                uri_str = f"{uri_str.rstrip('/')}/video"

    return uri_str


def _detect_connection_medium(source_type: str, uri: Union[str, int]) -> str:
    """Classify the physical or network connection medium."""
    st = source_type.lower()
    if st == "mobile":
        return "Mobile Browser (Push)"
    if st == "usb":
        return "Direct USB / V4L2"
    if st == "video_file":
        return "Local Video File"

    uri_str = str(uri).lower()
    if any(k in uri_str for k in ("192.168.42.", "192.168.43.", "192.168.137.", "10.42.0.", "127.0.0.1", "localhost")):
        return "USB Tethering / ADB"
    return "Wi-Fi Network"


def _check_host_reachable(uri: str, timeout_sec: float = 1.5) -> Tuple[bool, str]:
    """Fast non-blocking socket reachability probe to avoid 30s OpenCV hangs."""
    try:
        parsed = urlparse(uri)
        host = parsed.hostname
        port = parsed.port
        if not host:
            return True, "ok"
        if not port:
            port = 554 if parsed.scheme == "rtsp" else 80

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout_sec)
        res = s.connect_ex((host, port))
        s.close()
        if res == 0:
            return True, "ok"
        return False, f"Port {port} on {host} is unreachable (code {res})"
    except Exception as e:
        return False, f"Network probe failed: {str(e)}"


def list_system_cameras() -> List[Dict[str, Any]]:
    """Scan and list available USB/V4L2 camera devices WITHOUT opening them.
    
    Reads metadata from /sys/class/video4linux so the internal laptop webcam
    is never powered on or opened during device discovery.
    """
    devices = []
    v4l_paths = sorted(glob.glob("/dev/video*"))

    for path in v4l_paths:
        try:
            idx_str = path.replace("/dev/video", "")
            if idx_str.isdigit():
                idx = int(idx_str)
                name = ""
                sysfs_name = f"/sys/class/video4linux/video{idx}/name"
                try:
                    with open(sysfs_name, "r") as f:
                        name = f.read().strip()
                except OSError:
                    name = f"Video Device ({path})"

                is_internal = any(
                    kw in name.lower() for kw in ("integrated", "internal", "webcam", "isight", "camera hub")
                )
                devices.append({
                    "id": path,
                    "device_index": idx,
                    "name": f"{name} ({path})" if name else f"Camera Device ({path})",
                    "available": True,
                    "internal": is_internal,
                })
        except Exception:
            pass

    return devices


@dataclass
class CameraHealth:
    source_type: str  # "mobile", "rtsp", "http_mjpeg", "usb", "video_file"
    source_uri: str
    is_connected: bool
    status: str  # "connected", "reconnecting", "disconnected", "standby"
    fps: float
    last_frame_ts: float
    frame_count: int
    connection_medium: str = "Wi-Fi Network"
    reconnect_attempts: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "source_uri": str(self.source_uri),
            "connection_medium": self.connection_medium,
            "is_connected": self.is_connected,
            "status": self.status,
            "fps": round(self.fps, 1),
            "last_frame_ts": round(self.last_frame_ts, 3),
            "frame_count": self.frame_count,
            "reconnect_attempts": self.reconnect_attempts,
            "error_message": self.error_message,
        }


class CameraSource:
    """Thread-safe camera stream client supporting USB, RTSP, HTTP, Video, and Mobile Ingest."""

    def __init__(
        self,
        source_type: str = "mobile",
        source_uri: Union[str, int] = "browser",
        target_fps: int = 15,
        buffer_size: int = 5,
        connection_timeout_sec: float = 6.0,
    ):
        self.source_type = source_type.lower()
        self.source_uri = _normalize_uri(self.source_type, source_uri)
        self.connection_medium = _detect_connection_medium(self.source_type, self.source_uri)
        self.target_fps = target_fps
        self.connection_timeout_sec = connection_timeout_sec

        self.buffer = FrameBuffer(max_frames=buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Health & FPS tracking
        self._last_frame_ts = 0.0
        self._frame_counter = 0
        self._fps_window_start = time.time()
        self._fps_frames = 0
        self._current_fps = 0.0
        self._reconnect_attempts = 0
        self._last_error: Optional[str] = None
        self._status = "standby" if self.source_type == "mobile" else "disconnected"

    def start(self) -> None:
        """Start background worker if source is pull-based (usb/rtsp/http/video)."""
        with self._lock:
            if self._running:
                return
            self._running = True
            if self.source_type != "mobile":
                self._status = "reconnecting"
                self._thread = threading.Thread(
                    target=self._capture_worker_loop,
                    name="FaceAuthCameraWorker",
                    daemon=True,
                )
                self._thread.start()
            else:
                self._status = "standby"

    def stop(self) -> None:
        """Stop capture worker and release camera resources."""
        with self._lock:
            self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            self._status = "disconnected"

    def ingest_frame(self, frame: np.ndarray) -> None:
        """Receive a frame pushed from an external client (e.g. Mobile browser ingest)."""
        if frame is None or frame.size == 0:
            return
        now = time.time()
        self.buffer.update(frame)
        with self._lock:
            self._last_frame_ts = now
            self._frame_counter += 1
            self._fps_frames += 1
            self._status = "connected"
            self._last_error = None
            self._update_fps(now)

    def get_latest_frame(self) -> Optional[Tuple[float, np.ndarray]]:
        """Return (timestamp, frame) if available."""
        return self.buffer.get_latest()

    def get_health(self) -> CameraHealth:
        """Return comprehensive health and connectivity metrics."""
        now = time.time()
        with self._lock:
            self._update_fps(now)
            is_fresh = (now - self._last_frame_ts) <= self.connection_timeout_sec and self._last_frame_ts > 0
            if not is_fresh:
                if self.source_type == "mobile":
                    status = "standby" if self._frame_counter == 0 else "disconnected"
                else:
                    status = "reconnecting" if self._running else "disconnected"
            else:
                status = "connected"

            return CameraHealth(
                source_type=self.source_type,
                source_uri=str(self.source_uri),
                connection_medium=self.connection_medium,
                is_connected=is_fresh,
                status=status,
                fps=self._current_fps if is_fresh else 0.0,
                last_frame_ts=self._last_frame_ts,
                frame_count=self._frame_counter,
                reconnect_attempts=self._reconnect_attempts,
                error_message=self._last_error,
            )

    def _update_fps(self, now: float) -> None:
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_window_start = now

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        uri = self.source_uri
        try:
            if self.source_type == "usb":
                idx = int(uri) if (isinstance(uri, int) or (isinstance(uri, str) and uri.isdigit())) else uri
                if isinstance(idx, int):
                    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(idx)
                else:
                    cap = cv2.VideoCapture(str(idx), cv2.CAP_V4L2)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(str(idx))

                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                    return cap

            elif self.source_type in ("rtsp", "http_mjpeg", "video_file"):
                uri_str = str(uri)
                if self.source_type in ("rtsp", "http_mjpeg"):
                    reachable, reason = _check_host_reachable(uri_str, timeout_sec=1.5)
                    if not reachable:
                        with self._lock:
                            self._last_error = f"Unreachable: {reason}. Ensure phone IP Webcam / DroidCam is active."
                        return None

                    if self.source_type == "rtsp":
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|max_delay;500000"

                cap = cv2.VideoCapture(uri_str)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap
                else:
                    with self._lock:
                        self._last_error = f"Could not open stream from {uri_str}"
        except Exception as e:
            with self._lock:
                self._last_error = f"Open capture error: {str(e)}"
        return None

    def _capture_worker_loop(self) -> None:
        """Background continuous capture loop with exponential backoff reconnection."""
        backoff_sec = 1.0
        max_backoff_sec = 15.0

        while self._running:
            with self._lock:
                self._status = "reconnecting"
            cap = self._open_capture()

            if cap is None or not cap.isOpened():
                with self._lock:
                    self._reconnect_attempts += 1
                    if not self._last_error:
                        self._last_error = f"Failed to open source {self.source_uri}"
                time.sleep(backoff_sec)
                backoff_sec = min(max_backoff_sec, backoff_sec * 1.4)
                continue

            # Connected successfully
            backoff_sec = 1.0
            with self._lock:
                self._reconnect_attempts = 0
                self._status = "connected"
                self._last_error = None

            delay = 1.0 / max(1, self.target_fps)
            is_file = self.source_type == "video_file"

            while self._running and cap.isOpened():
                ret, frame = cap.read()
                now = time.time()

                if not ret or frame is None:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        with self._lock:
                            self._last_error = "Frame read returned empty"
                        break

                # Downscale excessively large frames (e.g. 4K) to 1280p for HD clarity on distant faces
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    frame = cv2.resize(frame, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)

                # Apply orientation correction so display + inference agree.
                frame = _apply_orientation(frame)

                self.buffer.update(frame)
                with self._lock:
                    self._last_frame_ts = now
                    self._frame_counter += 1
                    self._fps_frames += 1
                    self._update_fps(now)

                if is_file:
                    time.sleep(delay)

            try:
                cap.release()
            except Exception:
                pass

            if self._running:
                with self._lock:
                    self._status = "reconnecting"
                    self._reconnect_attempts += 1
                time.sleep(backoff_sec)
                backoff_sec = min(max_backoff_sec, backoff_sec * 1.4)


class CameraManager:
    """Manages the active camera source for Face Authorization."""

    def __init__(self):
        self._lock = threading.Lock()
        self.camera = CameraSource(source_type="mobile", source_uri="browser")

    def configure_camera(
        self,
        source_type: str,
        source_uri: Union[str, int],
        target_fps: int = 15,
    ) -> CameraHealth:
        """Reconfigure or start a camera stream source dynamically."""
        with self._lock:
            self.camera.stop()
            self.camera = CameraSource(
                source_type=source_type,
                source_uri=source_uri,
                target_fps=target_fps,
            )
            self.camera.start()
            return self.camera.get_health()

    def ingest_frame(self, frame: np.ndarray) -> CameraHealth:
        self.camera.ingest_frame(frame)
        return self.camera.get_health()

    def get_latest_frame(self) -> Optional[Tuple[float, np.ndarray]]:
        return self.camera.get_latest_frame()

    def get_health(self) -> CameraHealth:
        return self.camera.get_health()

    def stop(self) -> None:
        self.camera.stop()
