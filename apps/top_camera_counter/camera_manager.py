"""Camera Source Manager for Top Camera Counter.

Supports:
- USB / V4L2 Webcams (/dev/video0, /dev/video1, etc.)
- Mobile IP Webcam via USB Tethering or ADB
  - IP Webcam: http://127.0.0.1:8080/video
  - DroidCam: http://127.0.0.1:4747/video
- RTSP Streams (rtsp://...)
- Browser-based Mobile Ingest (/ingest/frame)

Features:
- Background capture with auto-reconnect
- Low latency capture
- Real-time health monitoring (FPS, resolution, connection status)
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


def _normalize_uri(source_type: str, uri: Union[str, int]) -> str:
    """Normalize input URI for HTTP MJPEG, RTSP, or USB sources."""
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
            uri_str = f"{uri_str.rstrip('/')}/video"

    return uri_str


def _detect_connection_medium(source_type: str, uri: Union[str, int]) -> str:
    """Classify the physical or network connection medium."""
    st = source_type.lower()
    if st == "mobile":
        return "Mobile Browser (Push)"
    if st == "usb":
        return "Direct USB / V4L2"

    uri_str = str(uri).lower()
    if any(k in uri_str for k in ("192.168.42.", "192.168.43.", "192.168.137.", "10.42.0.", "127.0.0.1", "localhost")):
        return "USB Tethering / ADB Wire"
    return "Wi-Fi Network / RTSP"


def _check_host_reachable(uri: str, timeout_sec: float = 1.5) -> Tuple[bool, str]:
    """Fast non-blocking socket reachability probe."""
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
        result = s.connect_ex((host, port))
        s.close()
        if result == 0:
            return True, "ok"
        return False, f"Port {port} unreachable on {host}"
    except Exception as e:
        return False, f"Network check failed: {e}"


def list_system_cameras() -> List[Dict[str, Any]]:
    """Enumerate Linux /dev/video* devices."""
    cameras = []
    dev_nodes = sorted(glob.glob("/dev/video*"))

    for dev_path in dev_nodes:
        try:
            idx = int(dev_path.replace("/dev/video", ""))
        except ValueError:
            continue

        name = "Generic Video Device"
        sysfs_name_path = f"/sys/class/video4linux/video{idx}/name"
        if os.path.exists(sysfs_name_path):
            try:
                with open(sysfs_name_path, "r") as f:
                    name = f.read().strip()
            except OSError:
                pass

        name_lower = name.lower()
        is_internal = any(
            kw in name_lower
            for kw in ("integrated", "internal", "laptop", "built-in", "facetime", "isight")
        )
        is_mobile = any(
            kw in name_lower
            for kw in ("android", "pixel", "samsung", "droidcam", "iriun", "phone", "mobile", "uvc")
        )

        cameras.append({
            "id": dev_path,
            "device_index": idx,
            "name": f"{name} ({dev_path})" if name else f"Camera Device ({dev_path})",
            "available": True,
            "internal": is_internal,
            "mobile_candidate": is_mobile or not is_internal,
        })

    return cameras


@dataclass
class CameraHealth:
    source_type: str = "mobile"
    uri: str = "browser_push"
    connected: bool = False
    connecting: bool = False
    fps: float = 0.0
    frame_count: int = 0
    last_frame_time: float = 0.0
    reconnect_attempts: int = 0
    error_message: Optional[str] = None
    width: int = 0
    height: int = 0
    medium: str = "Mobile Browser (Push)"

    def to_dict(self) -> Dict[str, Any]:
        age_sec = (time.time() - self.last_frame_time) if self.last_frame_time > 0 else -1.0
        is_alive = self.connected and (age_sec >= 0 and age_sec < 5.0)

        return {
            "source_type": self.source_type,
            "uri": self.uri,
            "connected": is_alive,
            "connecting": self.connecting,
            "fps": round(self.fps, 1),
            "frame_count": self.frame_count,
            "last_frame_age_sec": round(age_sec, 2) if age_sec >= 0 else None,
            "reconnect_attempts": self.reconnect_attempts,
            "error_message": self.error_message,
            "resolution": f"{self.width}x{self.height}" if self.width and self.height else "N/A",
            "medium": self.medium,
        }


class CameraSource:
    """Manages video acquisition from USB or network streams with auto-reconnect."""

    def __init__(self, buffer_size: int = 10):
        self.buffer = FrameBuffer(max_frames=buffer_size)
        self.health = CameraHealth()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._fps_window: List[float] = []
        self._fps_window_size = 8

    def connect(
        self,
        source_type: str,
        uri: Union[str, int],
        fps_target: int = 25,
        width: int = 640,
        height: int = 480,
        rotation: int = 0,
    ) -> Tuple[bool, str]:
        """Connect to a camera source."""
        self.disconnect()

        clean_uri = _normalize_uri(source_type, uri)
        medium = _detect_connection_medium(source_type, uri)

        if source_type in ("http_mjpeg", "rtsp"):
            reachable, reason = _check_host_reachable(clean_uri, timeout_sec=2.0)
            if not reachable:
                with self._lock:
                    self.health = CameraHealth(
                        source_type=source_type,
                        uri=clean_uri,
                        connected=False,
                        error_message=reason,
                        medium=medium,
                    )
                return False, reason

        with self._lock:
            self.health = CameraHealth(
                source_type=source_type,
                uri=clean_uri,
                connecting=True,
                medium=medium,
            )
            self._stop_event.clear()
            self._fps_window.clear()

        self._thread = threading.Thread(
            target=self._capture_worker,
            args=(source_type, clean_uri, fps_target, width, height, rotation),
            name="TopCameraWorker",
            daemon=True,
        )
        self._thread.start()

        time.sleep(0.3)
        with self._lock:
            if self.health.error_message and not self.health.connected:
                return False, self.health.error_message
            return True, "Connecting"

    def disconnect(self) -> None:
        """Stop capture and release device."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        with self._lock:
            self.health.connected = False
            self.health.connecting = False
            self.health.fps = 0.0

    def ingest_push_frame(self, frame: np.ndarray) -> None:
        """Ingest a frame sent via HTTP POST /ingest/frame (browser push)."""
        self.buffer.update(frame)

        now = time.time()
        with self._lock:
            self.health.source_type = "mobile"
            self.health.uri = "browser_push"
            self.health.medium = "Mobile Browser (Push)"
            self.health.connected = True
            self.health.last_frame_time = now
            self.health.frame_count = self.buffer.frame_count
            self.health.width = frame.shape[1]
            self.health.height = frame.shape[0]

            self._fps_window.append(now)
            if len(self._fps_window) > self._fps_window_size:
                self._fps_window.pop(0)
            if len(self._fps_window) >= 2:
                duration = self._fps_window[-1] - self._fps_window[0]
                if duration > 0:
                    self.health.fps = (len(self._fps_window) - 1) / duration

    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame from buffer."""
        return self.buffer.get()

    def _capture_worker(
        self,
        source_type: str,
        uri: str,
        fps_target: int,
        req_width: int,
        req_height: int,
        rotation: int = 0,
    ) -> None:
        """Background loop reading frames with auto-reconnect."""
        delay = 1.0 / max(1, fps_target)

        while not self._stop_event.is_set():
            cap: Optional[cv2.VideoCapture] = None
            try:
                if source_type == "usb":
                    try:
                        dev_idx = int(str(uri).replace("/dev/video", ""))
                    except ValueError:
                        dev_idx = 0
                    cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(dev_idx)
                else:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"
                    cap = cv2.VideoCapture(uri)

                if not cap or not cap.isOpened():
                    with self._lock:
                        self.health.connected = False
                        self.health.connecting = False
                        self.health.reconnect_attempts += 1
                        self.health.error_message = f"Cannot open camera: {uri}"
                    time.sleep(0.5)
                    continue

                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if source_type == "usb":
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_height)
                        cap.set(cv2.CAP_PROP_FPS, fps_target)
                except Exception:
                    pass

                with self._lock:
                    self.health.connected = True
                    self.health.connecting = False
                    self.health.error_message = None

                consecutive_failures = 0
                while not self._stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        consecutive_failures += 1
                        if consecutive_failures > 8:
                            with self._lock:
                                self.health.connected = False
                                self.health.error_message = "Stream dropped. Reconnecting..."
                            break
                        time.sleep(0.01)
                        continue

                    consecutive_failures = 0
                    now = time.time()

                    if rotation:
                        if rotation == 90:
                            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                        elif rotation == 180:
                            frame = cv2.rotate(frame, cv2.ROTATE_180)
                        elif rotation == 270:
                            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    h, w = frame.shape[:2]
                    if w != req_width or h != req_height:
                        frame = cv2.resize(frame, (req_width, req_height), interpolation=cv2.INTER_LINEAR)

                    self.buffer.update(frame)

                    with self._lock:
                        self.health.last_frame_time = now
                        self.health.frame_count += 1
                        self.health.width = frame.shape[1]
                        self.health.height = frame.shape[0]

                        self._fps_window.append(now)
                        if len(self._fps_window) > self._fps_window_size:
                            self._fps_window.pop(0)
                        if len(self._fps_window) >= 2:
                            dur = self._fps_window[-1] - self._fps_window[0]
                            if dur > 0:
                                self.health.fps = (len(self._fps_window) - 1) / dur

                    time.sleep(delay * 0.15)

            except Exception as e:
                with self._lock:
                    self.health.connected = False
                    self.health.error_message = f"Capture error: {e}"
                    self.health.reconnect_attempts += 1
                time.sleep(1.5)
            finally:
                if cap is not None:
                    cap.release()


class CameraManager:
    """Coordinator for camera feeds."""

    def __init__(self):
        self.camera = CameraSource()

    def get_status(self) -> Dict[str, Any]:
        return {
            "health": self.camera.health.to_dict(),
            "available_usb_cameras": list_system_cameras(),
        }
