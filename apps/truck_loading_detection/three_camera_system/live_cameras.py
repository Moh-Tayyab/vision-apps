"""Low-latency, per-camera live capture for RTSP streams and local devices."""

from __future__ import annotations

import collections
import logging
import os
import threading
import time
from typing import Any, Deque, Dict, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CAMERA_IDS = ("cam_top", "cam_side", "cam_flank")
RECONNECT_DELAY_SEC = 2.0
JPEG_QUALITY = 80


def parse_camera_source(source: str, camera_id: Optional[str] = None) -> Union[str, int]:
    """Convert numeric USB device strings to indices; preserve network URLs and video files."""
    normalized = str(source).strip()
    if normalized.isdecimal():
        return int(normalized)
    if normalized.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return normalized

    # Demo/sample presets: "sample", "demo", "preset", or empty with camera_id
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rec_sync_dir = os.path.join(app_dir, "recorded_video", "synchronized")
    if camera_id and (normalized.lower() in ("sample", "demo", "preset", "auto", "") or not normalized):
        cam_role = camera_id.replace("cam_", "")
        candidate = os.path.join(rec_sync_dir, f"scenario1_{cam_role}_synced.mp4")
        if os.path.isfile(candidate):
            return candidate

    # Check direct file or relative file paths
    for candidate in (
        normalized,
        os.path.join(app_dir, normalized),
        os.path.join(app_dir, "recorded_video", normalized),
        os.path.join(rec_sync_dir, normalized),
        os.path.join(rec_sync_dir, os.path.basename(normalized)),
    ):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise ValueError(
        f"Camera source '{source}' must be an RTSP/HTTP URL, numeric USB index (e.g. 0), or valid video file"
    )


class CameraReader:
    """Owns one VideoCapture and publishes its latest JPEG, raw frame buffer, and health state."""

    def __init__(self, camera_id: str, source: Union[str, int]) -> None:
        self.camera_id = camera_id
        self.source = source
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"live-{camera_id}",
            daemon=True,
        )
        self._status = "starting"
        self._last_error: Optional[str] = None
        self._latest_jpeg: Optional[bytes] = None
        self._last_frame_at: Optional[float] = None
        self._frame_count = 0
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._fps = 0.0
        self._last_fps_time = time.monotonic()
        self._frames_since_fps = 0
        # Rolling circular buffer of raw frames (monotonic_time, frame) for PTS jitter synchronization
        self._frame_buffer: Deque[Tuple[float, np.ndarray]] = collections.deque(maxlen=30)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        capture = self._capture
        if capture is not None:
            capture.release()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            age = (
                max(0.0, time.monotonic() - self._last_frame_at)
                if self._last_frame_at is not None
                else None
            )
            status = self._status
            if status == "connected" and age is not None and age > 5.0:
                status = "stale"
            return {
                "camera_id": self.camera_id,
                "status": status,
                "last_error": self._last_error,
                "width": self._width,
                "height": self._height,
                "fps": round(self._fps, 1),
                "frames_received": self._frame_count,
                "last_frame_age_sec": round(age, 2) if age is not None else None,
            }

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def latest_raw_frame(self) -> Optional[Tuple[float, np.ndarray]]:
        """Returns the most recent (monotonic_timestamp, bgr_frame) tuple."""
        with self._lock:
            if not self._frame_buffer:
                return None
            return self._frame_buffer[-1]

    def get_frame_near_timestamp(
        self, target_time: float, max_delta_sec: float = 0.060
    ) -> Optional[np.ndarray]:
        """Finds the frame closest to target_time within max_delta_sec tolerance."""
        with self._lock:
            if not self._frame_buffer:
                return None
            best_frame = None
            best_diff = float("inf")
            for ts, frame in self._frame_buffer:
                diff = abs(ts - target_time)
                if diff < best_diff and diff <= max_delta_sec:
                    best_diff = diff
                    best_frame = frame
            return best_frame

    def _set_status(self, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            self._status = status
            self._last_error = error

    def _run(self) -> None:
        while not self._stop.is_set():
            capture = None
            try:
                if isinstance(self.source, str) and self.source.lower().startswith(
                    ("rtsp://", "rtsps://", "http://", "https://")
                ):
                    capture = cv2.VideoCapture(
                        self.source,
                        cv2.CAP_FFMPEG,
                        [
                            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
                        ],
                    )
                else:
                    capture = cv2.VideoCapture(self.source)
                self._capture = capture
                if not capture.isOpened():
                    self._set_status("disconnected", "Unable to open camera source")
                    capture.release()
                    self._stop.wait(RECONNECT_DELAY_SEC)
                    continue

                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._set_status("connecting")

                # Detect if reading from a file source
                is_file = isinstance(self.source, str) and not self.source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
                src_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
                if src_fps <= 0 or src_fps > 120 or not np.isfinite(src_fps):
                    src_fps = 30.0
                frame_interval = 1.0 / src_fps

                while not self._stop.is_set():
                    t_frame_start = time.monotonic()
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        if is_file:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ok, frame = capture.read()
                        if not ok or frame is None:
                            self._set_status("reconnecting", "No frame received")
                            break
                    encoded, buffer = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                    )
                    if not encoded:
                        self._set_status("error", "Could not encode camera frame")
                        continue

                    now = time.monotonic()
                    with self._lock:
                        self._latest_jpeg = buffer.tobytes()
                        self._last_frame_at = now
                        self._frame_count += 1
                        self._frames_since_fps += 1
                        self._height, self._width = frame.shape[:2]
                        elapsed = now - self._last_fps_time
                        if elapsed >= 1.0:
                            self._fps = self._frames_since_fps / elapsed
                            self._frames_since_fps = 0
                            self._last_fps_time = now
                        self._status = "connected"
                        self._last_error = None
                        self._frame_buffer.append((now, frame))

                    if is_file:
                        elapsed_f = time.monotonic() - t_frame_start
                        rem = frame_interval - elapsed_f
                        if rem > 0:
                            self._stop.wait(rem)
            except Exception:
                logger.exception("Live capture failed for %s", self.camera_id)
                self._set_status("error", "Camera capture raised an exception")
            finally:
                if capture is not None:
                    capture.release()
                self._capture = None

            if not self._stop.is_set():
                self._stop.wait(RECONNECT_DELAY_SEC)

        self._set_status("stopped")


class LiveCameraManager:
    """Starts and stops the three configured camera readers with real-time stream synchronization."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._readers: Dict[str, CameraReader] = {}
        self._counting_thread: Optional[threading.Thread] = None
        self._counting_stop = threading.Event()
        self._pipeline: Optional[Any] = None
        self._counting_active: bool = False
        self._live_counts: Dict[str, Any] = {
            "loaded": 0,
            "returned": 0,
            "net": 0,
            "confirmed_loaded": 0,
            "confirmed_returned": 0,
            "review_required": 0,
            "total_ticks": 0,
            "processing_fps": 0.0,
        }
        self._live_fps_time = time.monotonic()
        self._live_ticks_since_fps = 0
        self._live_fps = 0.0

    def start(self, sources: Dict[str, Union[str, int]]) -> None:
        if set(sources) != set(CAMERA_IDS):
            raise ValueError("Exactly cam_top, cam_side, and cam_flank sources are required")
        self.stop()
        readers = {
            camera_id: CameraReader(camera_id, source)
            for camera_id, source in sources.items()
        }
        with self._lock:
            self._readers = readers
        for reader in readers.values():
            reader.start()

    def stop(self) -> None:
        self.stop_counting()
        with self._lock:
            readers = list(self._readers.values())
            self._readers = {}
        for reader in readers:
            reader.stop()

    def get_synchronized_triplet(
        self, max_jitter_sec: float = 0.150
    ) -> Optional[Tuple[float, Dict[str, np.ndarray]]]:
        """
        PTS / Wall-clock Jitter Buffer: aligns incoming frames across all 3 cameras
        within a bounded jitter window (default 150ms). Returns (ref_time, frame_dict) or None.
        """
        with self._lock:
            readers = dict(self._readers)

        if len(readers) < len(CAMERA_IDS):
            return None

        # Gather latest raw frame timestamp from each reader
        latest_frames: Dict[str, Tuple[float, np.ndarray]] = {}
        for cid, reader in readers.items():
            entry = reader.latest_raw_frame()
            if entry is not None:
                latest_frames[cid] = entry

        # Wait until all cameras have produced at least one frame
        if len(latest_frames) < len(readers):
            return None

        timestamps = [ts for ts, _ in latest_frames.values()]
        ref_time = min(timestamps)

        triplet: Dict[str, np.ndarray] = {}
        for cid, reader in readers.items():
            f = reader.get_frame_near_timestamp(ref_time, max_delta_sec=max_jitter_sec)
            if f is None:
                # If within jitter tolerance of reference time, use latest frame
                latest_ts, latest_f = latest_frames[cid]
                if abs(latest_ts - ref_time) <= max_jitter_sec:
                    f = latest_f
            if f is None:
                return None
            triplet[cid] = f

        return (ref_time, triplet)

    def start_counting(self, pipeline: Any, target_fps: float = 20.0) -> None:
        """Starts background live carton counting across the synchronized streams."""
        with self._lock:
            if self._counting_active:
                return
            self._pipeline = pipeline
            self._counting_active = True
            self._counting_stop.clear()
            self._counting_thread = threading.Thread(
                target=self._counting_worker,
                args=(float(target_fps),),
                name="live-counting-worker",
                daemon=True,
            )
            self._counting_thread.start()
        logger.info("Live carton counting started at target %.1f FPS", target_fps)

    def stop_counting(self) -> None:
        """Stops background live counting worker cleanly."""
        thread = None
        with self._lock:
            if not self._counting_active:
                return
            self._counting_active = False
            self._counting_stop.set()
            thread = self._counting_thread
            self._counting_thread = None
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        logger.info("Live carton counting stopped")

    def _counting_worker(self, target_fps: float) -> None:
        interval = 1.0 / max(1.0, target_fps)
        tick_idx = 0
        while not self._counting_stop.is_set():
            t_start = time.monotonic()
            synced = self.get_synchronized_triplet(max_jitter_sec=0.150)
            if synced is not None:
                ref_time, frame_dict = synced
                tick_idx += 1
                try:
                    tick_output = self._pipeline.process_synchronized_frames(
                        frames=frame_dict,
                        frame_idx=tick_idx,
                        override_time_sec=ref_time,
                    )
                    now = time.monotonic()
                    with self._lock:
                        self._live_ticks_since_fps += 1
                        elapsed_fps = now - self._live_fps_time
                        if elapsed_fps >= 1.0:
                            self._live_fps = self._live_ticks_since_fps / elapsed_fps
                            self._live_ticks_since_fps = 0
                            self._live_fps_time = now

                        if not getattr(self._pipeline, "require_calibrated_fusion", False):
                            cur_loaded = self._pipeline.total_in
                            cur_returned = self._pipeline.total_out
                        else:
                            cur_loaded = self._pipeline.confirmed_in
                            cur_returned = self._pipeline.confirmed_out

                        self._live_counts = {
                            "loaded": cur_loaded,
                            "returned": cur_returned,
                            "net": cur_loaded - cur_returned,
                            "confirmed_loaded": cur_loaded,
                            "confirmed_returned": cur_returned,
                            "review_required": self._pipeline.review_required_count,
                            "total_ticks": tick_idx,
                            "processing_fps": round(self._live_fps, 1),
                            "active_tracklets": getattr(tick_output, "active_tracklets", {}),
                            "timestamp_sec": round(ref_time, 2),
                        }
                except Exception:
                    logger.exception("Error during live inference tick %d", tick_idx)

            elapsed = time.monotonic() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                self._counting_stop.wait(sleep_time)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            readers = dict(self._readers)
            counting_active = self._counting_active
            live_counts = dict(self._live_counts)

        cameras = {
            camera_id: (
                readers[camera_id].status()
                if camera_id in readers
                else {
                    "camera_id": camera_id,
                    "status": "not_configured",
                    "last_error": None,
                    "width": None,
                    "height": None,
                    "fps": 0.0,
                    "frames_received": 0,
                    "last_frame_age_sec": None,
                }
            )
            for camera_id in CAMERA_IDS
        }
        connected = sum(camera["status"] == "connected" for camera in cameras.values())
        ready = connected == len(CAMERA_IDS)

        counting_blocker = (
            None
            if counting_active
            else (
                "Live counting is not implemented; verified calibration is also required for all three cameras"
            )
        )

        return {
            "active": bool(readers),
            "ready": ready,
            "connected_cameras": connected,
            "required_cameras": len(CAMERA_IDS),
            "cameras": cameras,
            "counting_enabled": counting_active,
            "counting_active": counting_active,
            "counting_blocker": counting_blocker,
            "live_counts": live_counts,
        }

    def latest_jpeg(self, camera_id: str) -> Optional[bytes]:
        with self._lock:
            reader = self._readers.get(camera_id)
        return reader.latest_jpeg() if reader else None

    def capture_evidence_collage(
        self,
        title: str = "SYNCHRONIZED AUDIT COLLAGE",
        subtitle: str = "",
        is_discrepancy: bool = False,
        output_path: Optional[str] = None,
        custom_frames: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[str]:
        """
        Renders and persists a synchronized 3-camera audit snapshot collage.
        Overlays timestamp, HUD banner, and discrepancy status.
        """
        from pathlib import Path

        now = time.time()
        frames: Dict[str, np.ndarray] = {}

        if custom_frames:
            frames = dict(custom_frames)
        else:
            with self._lock:
                readers = dict(self._readers)
            for cid in CAMERA_IDS:
                if cid in readers:
                    raw = readers[cid].latest_raw_frame()
                    if raw is not None:
                        frames[cid] = raw[1]

        tile_w, tile_h = 480, 270
        panes = []

        for cid in CAMERA_IDS:
            if cid in frames and isinstance(frames[cid], np.ndarray) and frames[cid].size > 0:
                pane = cv2.resize(frames[cid], (tile_w, tile_h))
            else:
                pane = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                cv2.putText(
                    pane,
                    f"[{cid.upper()} OFFLINE]",
                    (tile_w // 4, tile_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (120, 120, 120),
                    2,
                )
            cv2.rectangle(pane, (0, 0), (tile_w, 28), (20, 20, 20), -1)
            cv2.putText(
                pane,
                f"CAM: {cid.upper()}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
            )
            panes.append(pane)

        triplet_canvas = np.hstack(panes)
        total_w = tile_w * len(CAMERA_IDS)
        header_h = 36
        footer_h = 36
        full_h = tile_h + header_h + footer_h

        collage = np.zeros((full_h, total_w, 3), dtype=np.uint8)

        # Header banner (dark slate)
        cv2.rectangle(collage, (0, 0), (total_w, header_h), (35, 35, 35), -1)
        header_text = f"{title} | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}"
        cv2.putText(
            collage,
            header_text,
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # Body
        collage[header_h : header_h + tile_h, 0:total_w] = triplet_canvas

        # Footer banner
        footer_color = (0, 0, 180) if is_discrepancy else (0, 140, 50)
        cv2.rectangle(collage, (0, full_h - footer_h), (total_w, full_h), footer_color, -1)
        status_label = "[DISCREPANCY ALERT]" if is_discrepancy else "[AUDIT VERIFIED]"
        footer_text = f"{status_label} {subtitle}".strip()
        cv2.putText(
            collage,
            footer_text,
            (15, full_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        import tempfile

        if not output_path:
            app_dir = Path(__file__).resolve().parent.parent
            candidate_dir = app_dir / "output_videos" / "audit_evidence"
            try:
                candidate_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(candidate_dir / f"audit_scale_{int(now * 1000)}.jpg")
            except OSError:
                fallback_dir = Path(tempfile.gettempdir()) / "audit_evidence"
                fallback_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(fallback_dir / f"audit_scale_{int(now * 1000)}.jpg")
        else:
            try:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                output_path = str(Path(tempfile.gettempdir()) / Path(output_path).name)
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), collage, [cv2.IMWRITE_JPEG_QUALITY, 85])
        logger.info("Saved synchronized audit evidence collage to %s", output_path)
        return str(output_path)

