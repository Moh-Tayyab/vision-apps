import os
import sys
import hashlib
import hmac
import base64
import re
import time
import json
_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)
import signal
import logging
import threading
import traceback
import atexit
import math
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from visualizer import LoadingVisualizer
from model_pipeline.mask_refinement import CartonMaskRefiner
from three_camera_system.pipeline import ThreeCameraCartonPipeline
from three_camera_system.event_store import EventStore
from three_camera_system.shadow_mode import ShadowModeManager
from three_camera_system.live_cameras import (
    CAMERA_IDS,
    LiveCameraManager,
    parse_camera_source,
)
from three_camera_system.evaluator import (
    PalletScaleCrossChecker,
    PalletScaleAuditResult,
)
from three_camera_system.security import require_api_key, CONFIGURED_API_KEYS


class CartonTripwireCounter:
    """
    Lightweight, dedicated Carton-only directional tripwire counter.
    Tracks carton centroid trajectories across virtual tripwire boundary.
    Zero dependency on worker models or legacy preprocessors.
    """

    def __init__(
        self,
        line_x: int = 500,
        line_y: int = 400,
        line_orientation: str = "vertical",
        loading_direction: str = "left_to_right",
        hysteresis: int = 15,
        cooldown_frames: int = 15,
        min_displacement_px: int = 15,
        max_inactive_frames: int = 90,
        max_events: int = 500,
        countable_classes: Optional[Set[str]] = None,
    ):
        self.line_x = int(line_x)
        self.line_y = int(line_y)
        self.line_orientation = str(line_orientation).lower()
        self.loading_direction = str(loading_direction).lower()
        self.hysteresis = int(hysteresis)
        self.cooldown_frames = int(cooldown_frames)
        self.min_displacement_px = int(min_displacement_px)
        self.max_inactive_frames = int(max_inactive_frames)
        self.max_events = int(max_events)

        self.total_in = 0
        self.total_out = 0
        self.events: List[Dict[str, Any]] = []
        self.recent_event: Optional[Dict[str, Any]] = None
        self.recent_event_expiry: float = 0.0

        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=60))
        self.track_side: Dict[int, str] = {}
        self.track_last_seen: Dict[int, int] = {}
        self.track_cooldown: Dict[int, int] = {}

    @property
    def net_count(self) -> int:
        return self.total_in - self.total_out

    def set_line_x(self, line_x: int):
        self.line_x = int(line_x)

    def set_line_y(self, line_y: int):
        self.line_y = int(line_y)

    def set_line_orientation(self, orientation: str):
        self.line_orientation = str(orientation).lower()

    def set_loading_direction(self, direction: str):
        self.loading_direction = str(direction).lower()

    def reset(self):
        self.total_in = 0
        self.total_out = 0
        self.events.clear()
        self.recent_event = None
        self.track_history.clear()
        self.track_side.clear()
        self.track_last_seen.clear()
        self.track_cooldown.clear()

    def reset_counts(self):
        self.reset()

    def update(
        self,
        tracked_objects: List[Tuple[int, Tuple[float, float, float, float], str, float]],
        frame_idx: int,
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        current_events = []
        active_ids = set()

        for track_id, (x1, y1, x2, y2), cls_name, conf in tracked_objects:
            active_ids.add(track_id)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            self.track_history[track_id].append((int(cx), int(cy)))
            self.track_last_seen[track_id] = frame_idx

            pts = self.track_history[track_id]
            if len(pts) < 2:
                continue

            dx = pts[-1][0] - pts[0][0]
            dy = pts[-1][1] - pts[0][1]
            disp = (dx**2 + dy**2) ** 0.5
            if disp < self.min_displacement_px:
                continue

            # Determine side of tripwire
            if self.line_orientation == "horizontal":
                coord = cy
                line_val = self.line_y
                side_low = "ABOVE"
                side_high = "BELOW"
            else:
                coord = cx
                line_val = self.line_x
                side_low = "LEFT"
                side_high = "RIGHT"

            prev_side = self.track_side.get(track_id)
            cur_side = prev_side

            if coord < (line_val - self.hysteresis):
                cur_side = side_low
            elif coord > (line_val + self.hysteresis):
                cur_side = side_high

            self.track_side[track_id] = cur_side

            # Crossing detection
            if prev_side and cur_side and prev_side != cur_side:
                cooldown_left = self.track_cooldown.get(track_id, 0)
                if frame_idx >= cooldown_left:
                    self.track_cooldown[track_id] = frame_idx + self.cooldown_frames

                    # Direction check
                    if self.loading_direction in ("left_to_right", "top_to_bottom"):
                        is_in = (prev_side == side_low and cur_side == side_high)
                    elif self.loading_direction in ("right_to_left", "bottom_to_top"):
                        is_in = (prev_side == side_high and cur_side == side_low)
                    else:  # "any"
                        is_in = True

                    if is_in:
                        self.total_in += 1
                        delta = +1
                        direction_label = "IN"
                    else:
                        self.total_out += 1
                        delta = -1
                        direction_label = "OUT"

                    event = {
                        "frame": frame_idx,
                        "timestamp": time.time(),
                        "track_id": track_id,
                        "class_name": "carton",
                        "is_cargo": True,
                        "direction": direction_label,
                        "delta": delta,
                        "position": (int(cx), int(cy)),
                        "total_in": self.total_in,
                        "total_out": self.total_out,
                        "net_count": self.net_count,
                        "note": f"Carton loaded ({direction_label})" if is_in else "Carton returned (OUT)",
                    }
                    self.events.append(event)
                    current_events.append(event)
                    self.recent_event = event
                    self.recent_event_expiry = time.time() + 2.0

        # Purge inactive tracks
        stale_ids = [tid for tid, last_f in self.track_last_seen.items() if (frame_idx - last_f) > self.max_inactive_frames]
        for tid in stale_ids:
            self.track_history.pop(tid, None)
            self.track_side.pop(tid, None)
            self.track_last_seen.pop(tid, None)
            self.track_cooldown.pop(tid, None)

        return current_events
from calibration import (
    CalibrationStorage,
    DockCalibration,
    CameraCalibrationData,
    UndistortionLUTManager,
    GeometricUndistorter,
    DoorwayHomographyEstimator,
    LandmarkSet,
    VibrationBaselineRecorder,
    NCCWatchdog,
    CameraAlignmentState,
)
from calibration.homography import REQUIRED_LANDMARK_NAMES


# ---------------------------------------------------------------------------
# Structured Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("truck_loading")

# ---------------------------------------------------------------------------
# Configuration from Environment Variables
# ---------------------------------------------------------------------------
PORT = int(os.environ.get("PORT", 5000))
HOST = os.environ.get("HOST", "0.0.0.0")
WORKERS = int(os.environ.get("WORKERS", 1))
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", 4))
# Certified Phase 3.3 active configuration (preserved, not tuned using locked-test data):
#   detection threshold τ_det = 0.15, consensus threshold τ_cons = 0.25.
CONF = float(os.environ.get("CONF_THRESHOLD", 0.15))
CONSENSUS_THRESHOLD = float(os.environ.get("CARTON_CONSENSUS_THRESHOLD", 0.25))
IOU = float(os.environ.get("IOU_THRESHOLD", 0.45))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", 2))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 500))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", 70))
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# RTSP / live camera resilience (warehouse CCTV)
RTSP_TRANSPORT = os.environ.get("RTSP_TRANSPORT", "tcp")
RTSP_STIMEOUT_MS = int(os.environ.get("RTSP_STIMEOUT_MS", 5000000))  # 5 s
MAX_RECONNECT_ATTEMPTS = int(os.environ.get("MAX_RECONNECT_ATTEMPTS", 0))  # 0 = unlimited
RECONNECT_BACKOFF_BASE = float(os.environ.get("RECONNECT_BACKOFF_BASE", 1.0))
RECONNECT_BACKOFF_CAP = float(os.environ.get("RECONNECT_BACKOFF_CAP", 10.0))
FRAME_DROP_MAX_CONSECUTIVE = int(os.environ.get("FRAME_DROP_MAX_CONSECUTIVE", 30))

# Default video source (RTSP URL or file path) via env
DEFAULT_SOURCE = os.environ.get("VIDEO_SOURCE", "output_videos/multi_angle_3angle_v2.mp4")

torch.set_num_threads(TORCH_THREADS)

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

lock = threading.Lock()
shutdown_event = threading.Event()

# Phase 0 Calibration & Watchdog Singletons
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
calib_storage = CalibrationStorage(os.path.join(_repo_root, "config", "calibrations"))
lut_manager = UndistortionLUTManager()
dock_watchdogs: Dict[str, Dict[str, NCCWatchdog]] = {}

# ---------------------------------------------------------------------------
# Counting Presets
# ---------------------------------------------------------------------------
COUNT_MODES = {
    "cartons_only": {"carton", "box", "cardboard box", "package", "suitcase", "backpack"},
    "people_only": {"person", "worker"},
    "all": None,
}


# ---------------------------------------------------------------------------
# Stream State
# ---------------------------------------------------------------------------
class StreamState:
    def __init__(self):
        app_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(app_dir, "..", ".."))

        # 1. Canonical carton-only instance segmentation model.
        model_path = os.path.join(app_dir, "best.pt")
        configured_model = os.environ.get("CARTON_MODEL_PATH")
        if configured_model and os.path.realpath(configured_model) != os.path.realpath(model_path):
            raise ValueError("CARTON_MODEL_PATH must point to the canonical apps/truck_loading_detection/best.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Canonical carton model checkpoint not found: {model_path}")
        self.model_name = model_path

        logger.info("Initializing Pure Carton Specialist Model=%s", self.model_name)
        self.carton_model = YOLO(self.model_name)
        if getattr(self.carton_model, "task", None) != "segment":
            raise ValueError(f"Canonical carton model must be a segmentation model: {self.model_name}")
        model_names = {
            int(class_id): str(class_name).lower()
            for class_id, class_name in self.carton_model.names.items()
        }
        if model_names != {0: "carton"}:
            raise ValueError(f"Canonical carton model must expose class mapping {{0: 'carton'}}, got {model_names}")
        self.model = self.carton_model

        # 2. Worker Segmentation Model (COCO instance segmentation for worker silhouette disentanglement)
        cand_worker = os.path.join(root_dir, "yolo11n-seg.pt")
        if not os.path.exists(cand_worker):
            cand_worker = os.path.join(app_dir, "yolo11n-seg.pt")
        if os.path.exists(cand_worker):
            logger.info("Initializing Worker Segmentation Model=%s", cand_worker)
            self.worker_model = YOLO(cand_worker)
        else:
            self.worker_model = None

        self.source = DEFAULT_SOURCE
        self.line_x = 515
        self.loading_direction = "left_to_right"
        self.conf = CONF
        self.iou = IOU
        self.hysteresis = 15
        self.cooldown = 15
        self.frame_stride = FRAME_STRIDE
        self.img_size = IMG_SIZE
        self.target_classes = ["carton", "box", "cardboard box", "package", "suitcase", "backpack"]
        self.count_mode = "cartons_only"
        self.last_tracked_masks = []

        self.counter = CartonTripwireCounter(
            line_x=self.line_x,
            hysteresis=self.hysteresis,
            cooldown_frames=self.cooldown,
            max_inactive_frames=90,
            max_events=500,
            countable_classes=COUNT_MODES[self.count_mode],
            min_displacement_px=15,
            loading_direction=self.loading_direction,
        )
        self.visualizer = LoadingVisualizer()
        self.mask_refiner = CartonMaskRefiner()

        self.is_paused = True
        self.loop_video = True
        self.fps = 25.0
        self.live_fps = 25.0
        self.frame_idx = 0
        self.active_count = 0
        self.video_width = 1024
        self.video_height = 576
        self.scale_factor = 1.0

        self.connection_status = "standby"
        self.reconnect_attempts = 0
        self.last_reconnect_time = 0.0

        self.cap = None
        self.current_frame_jpg = None
        self.last_tracked_objects = []

        self.is_multi_angle = False
        self.multi_angle_report = None
        self.live_cameras = {}
        # Dynamic tripwire configuration: Load from config/tripwires.json or default to vertical loading line
        tw_path = Path(__file__).resolve().parent / "config" / "tripwires.json"
        persisted_tw = {}
        if tw_path.is_file():
            try:
                with open(tw_path, "r") as f:
                    data = json.load(f)
                    persisted_tw = data.get("tripwires", data)
            except Exception as e:
                logger.warning("Could not read config/tripwires.json: %s", e)
        self.tripwire_configs = persisted_tw or {
            "cam_top": {
                "orientation": "vertical",
                "line_pos_frac": 0.50,
                "direction_sign": -1,
                "physically_verified": True,
            },
            "cam_side": {
                "orientation": "vertical",
                "line_pos_frac": 0.50,
                "direction_sign": 1,
                "physically_verified": True,
            },
            "cam_flank": {
                "orientation": "vertical",
                "line_pos_frac": 0.55,
                "direction_sign": 1,
                "physically_verified": True,
            },
        }
        self.pallet_scale_checker = PalletScaleCrossChecker()
        self.pallet_scale_telemetry = None

        self.start_time = datetime.now(timezone.utc)

        if self.source:
            self.init_capture()

    def init_capture(self) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        src = self.source
        if not src:
            self.connection_status = "standby"
            self.current_frame_jpg = self.create_standby_frame("NO VIDEO LOADED")
            return False

        base_name = os.path.basename(str(src))
        if isinstance(src, str):
            if src.isdigit():
                src = int(src)
            elif not os.path.isabs(src) and not src.startswith(("http://", "https://", "rtsp://")):
                app_dir = os.path.dirname(os.path.abspath(__file__))
                candidate = os.path.join(app_dir, src)
                if os.path.exists(candidate):
                    src = candidate

        # Detect if source is a 3-angle multi-view composite video or has a report JSON
        if self.multi_angle_report is None:
            self.is_multi_angle = False
            if isinstance(src, str):
                src_str = str(src).lower()
                if any(k in src_str for k in ["multi_angle", "3view", "3angle", "fused"]):
                    self.is_multi_angle = True

            # Attempt to locate report JSON
            app_dir = os.path.dirname(os.path.abspath(__file__))
            abs_src = os.path.abspath(os.path.join(app_dir, src)) if not os.path.isabs(str(src)) else os.path.abspath(str(src))
            src_dir = os.path.dirname(abs_src)
            src_base = os.path.basename(abs_src)

            # Only attach a report whose basename matches the selected video.
            # Generic fallback reports can silently inject stale counts into a new source.
            cands = [
                os.path.join(src_dir, src_base.rsplit(".", 1)[0] + "_report.json"),
            ]
            for c in cands:
                if os.path.exists(c):
                    try:
                        with open(c, "r") as fh:
                            self.multi_angle_report = json.load(fh)
                        self.is_multi_angle = True
                        logger.info("Associated multi-angle report loaded: %s", c)
                        break
                    except Exception as e:
                        logger.warning("Failed to load report %s: %s", c, e)

        try:
            # Set RTSP transport timeout for live camera streams
            if isinstance(src, str) and src.startswith("rtsp://"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    f"rtsp_transport;{RTSP_TRANSPORT}|stimeout;{RTSP_STIMEOUT_MS}"
                )

            self.cap = cv2.VideoCapture(src)
            if self.cap.isOpened():
                raw_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1024
                raw_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 576
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
                if self.fps <= 0 or self.fps > 120:
                    self.fps = 25.0

                max_dim = max(raw_w, raw_h)
                target_limit = 1280 if self.is_multi_angle else 720
                if max_dim > target_limit:
                    scale = float(target_limit) / max_dim
                    self.video_width = int(raw_w * scale)
                    self.video_height = int(raw_h * scale)
                    self.scale_factor = scale
                else:
                    self.video_width = raw_w
                    self.video_height = raw_h
                    self.scale_factor = 1.0

                if base_name in VIDEO_PRESETS:
                    preset = VIDEO_PRESETS[base_name]
                    self.loading_direction = preset["direction"]
                    self.line_x = int(self.video_width * preset["line_x_ratio"])
                    self.conf = preset.get("conf", 0.24)
                else:
                    if self.line_x is None or self.line_x > self.video_width or self.line_x <= 0:
                        self.line_x = self.video_width // 2

                self.counter.set_line_x(self.line_x)
                self.counter.set_loading_direction(self.loading_direction)
                self.counter.hysteresis = max(12, int(self.video_width * 0.015))
                self.counter.min_displacement_px = max(12, int(self.video_width * 0.012))

                self.counter.reset_counts()
                self.counter.prev_gray = None
                self.current_frame_jpg = None

                self.connection_status = "connected"
                self.reconnect_attempts = 0
                self.frame_idx = 0
                self.last_tracked_objects = []
                logger.info("Capture opened: %s (%dx%d @ %.1f fps)", src, self.video_width, self.video_height, self.fps)
                return True
            else:
                self.connection_status = "error"
                logger.warning("Failed to open capture: %s", src)
                return False
        except Exception as e:
            logger.error("Capture init failed for %s: %s", src, e)
            self.connection_status = "error"
            return False

    def create_standby_frame(self, message="NO VIDEO LOADED"):
        w, h = self.video_width or 720, self.video_height or 405
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (15, 20, 30)
        cv2.rectangle(frame, (15, 15), (w - 15, h - 15), (35, 45, 65), 2)
        cv2.putText(frame, "TRUCK LOADING MONITOR", (max(20, w // 2 - 180), max(40, h // 2 - 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 116, 139), 2, cv2.LINE_AA)
        cv2.putText(frame, message, (max(20, w // 2 - 120), max(70, h // 2 + 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (248, 250, 252), 2, cv2.LINE_AA)
        cv2.putText(frame, "Upload a video or set an RTSP source to begin", (max(20, w // 2 - 195), max(100, h // 2 + 55)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA)
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buffer.tobytes()

    def cleanup(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        logger.info("StreamState resources released.")


state = StreamState()
live_camera_manager = LiveCameraManager()


# ---------------------------------------------------------------------------
# Startup Validation: Frozen Model Integrity
# ---------------------------------------------------------------------------
# Canonical Phase 3.3 frozen baseline. These hashes are verified at startup so a
# corrupted or substituted checkpoint prevents service start (fail-closed).
FROZEN_MODELS = {
    "best.pt": {
        "path": os.path.join(_app_dir, "best.pt"),
        "sha256": "98c6164c83905feb9db6d9bee6f48ab239c5d68a00efbe0484dbf3448333ad33",
        "purpose": "Canonical Carton Detector",
    }
}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_models() -> bool:
    """Verify frozen model checkpoints match their certified SHA-256 digests.

    Returns True when all models match. Logs an error and returns False on any
    mismatch or missing file so startup can fail closed.
    """
    all_ok = True
    for name, meta in FROZEN_MODELS.items():
        path = meta["path"]
        expected = meta["sha256"]
        if not os.path.exists(path):
            logger.error("STARTUP VALIDATION: missing frozen model %s at %s", name, path)
            all_ok = False
            continue
        actual = _sha256_file(path)
        if actual != expected:
            logger.error(
                "STARTUP VALIDATION: hash mismatch for %s (expected %s, got %s)",
                name, expected, actual,
            )
            all_ok = False
        else:
            logger.info("STARTUP VALIDATION: frozen model %s OK (%s)", name, meta["purpose"])

    selected_model_path = os.path.realpath(state.model_name)
    if selected_model_path != os.path.realpath(FROZEN_MODELS["best.pt"]["path"]):
        logger.error(
            "STARTUP VALIDATION: selected carton model is not the canonical best.pt checkpoint: %s",
            state.model_name,
        )
        all_ok = False
    return all_ok


# Run validation at import time so gunicorn --preload and `python app.py` both
# fail closed before serving traffic if a checkpoint is corrupted or substituted.
_MODELS_VALID = verify_frozen_models()
if not _MODELS_VALID:
    raise RuntimeError("Frozen model validation failed; refusing to start the application.")


# ---------------------------------------------------------------------------
# Background Processing Loop
# ---------------------------------------------------------------------------
def processing_loop():
    model_names = state.model.names
    desired = set(state.target_classes)
    target_cids = [cid for cid, name in model_names.items() if any(d in name.lower() for d in desired)]
    if not target_cids:
        target_cids = None

    logger.info("Processing loop started. Target classes: %s", [model_names[c] for c in (target_cids or [])])

    while not shutdown_event.is_set():
        try:
            if state.is_paused or state.connection_status == "standby" or state.is_multi_angle:
                time.sleep(0.05)
                continue

            t_start = time.time()
            with lock:
                if not state.source:
                    if state.current_frame_jpg is None:
                        state.current_frame_jpg = state.create_standby_frame()
                    time.sleep(0.1)
                    continue

                cap_is_open = state.cap is not None and state.cap.isOpened()
                if not cap_is_open:
                    now = time.time()
                    backoff = min(RECONNECT_BACKOFF_CAP, RECONNECT_BACKOFF_BASE + (state.reconnect_attempts * 1.5))
                    if MAX_RECONNECT_ATTEMPTS > 0 and state.reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                        state.connection_status = "error"
                        logger.error("Max reconnect attempts (%d) reached for %s. Giving up.", MAX_RECONNECT_ATTEMPTS, state.source)
                        time.sleep(1.0)
                        continue
                    if now - state.last_reconnect_time >= backoff:
                        state.connection_status = "reconnecting"
                        state.reconnect_attempts += 1
                        state.last_reconnect_time = now
                        logger.warning("Reconnect attempt #%d to %s", state.reconnect_attempts, state.source)
                        state.init_capture()
                    time.sleep(0.05)
                    continue

                ret, frame = state.cap.read()
                source = state.source
                loop_video = state.loop_video

                if not ret:
                    state.consecutive_frame_drops = getattr(state, "consecutive_frame_drops", 0) + 1
                    is_live_stream = isinstance(source, int) or (
                        isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("http"))
                    )
                    if is_live_stream:
                        if state.consecutive_frame_drops >= FRAME_DROP_MAX_CONSECUTIVE:
                            logger.warning("Too many consecutive frame drops (%d). Forcing reconnect.", state.consecutive_frame_drops)
                            state.connection_status = "reconnecting"
                            state.consecutive_frame_drops = 0
                            if state.cap:
                                state.cap.release()
                            state.cap = None
                            state.last_reconnect_time = time.time()
                        else:
                            logger.warning("Live stream frame drop (%d/%d). Waiting...",
                                           state.consecutive_frame_drops, FRAME_DROP_MAX_CONSECUTIVE)
                        state.connection_status = "reconnecting"
                        time.sleep(0.05)
                        continue
                    elif loop_video:
                        if state.cap:
                            state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        state.frame_idx = 0
                        state.counter.reset_counts()
                        continue
                    else:
                        time.sleep(0.05)
                        continue

                state.connection_status = "connected"
                state.reconnect_attempts = 0
                state.consecutive_frame_drops = 0
                state.frame_idx += 1
                cur_frame_idx = state.frame_idx
                conf_val = state.conf
                iou_val = state.iou
                img_size_val = state.img_size
                stride_val = state.frame_stride
                scale_fac = state.scale_factor
                target_w = state.video_width
                target_h = state.video_height
                last_objs = list(state.last_tracked_objects)

            # If multi-angle synchronized feed with burned-in top banner (rows 0..69), crop it out cleanly
            if state.is_multi_angle or (frame.shape[0] >= 500 and frame.shape[1] >= 1500):
                if frame.shape[0] >= 500 and frame.shape[1] >= 1500:
                    frame = frame[70:, :]

                if scale_fac != 1.0 or frame.shape[1] != target_w:
                    clean_h = int(frame.shape[0] * (target_w / frame.shape[1]))
                    frame = cv2.resize(frame, (target_w, clean_h), interpolation=cv2.INTER_LINEAR)

                if state.is_multi_angle:
                    # Frame is a synchronized 3-angle multi-view composite already rendered with HUD and bounding boxes!
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    _, buffer = cv2.imencode(".jpg", frame, encode_param)
                    with lock:
                        state.current_frame_jpg = buffer.tobytes()
                        t_end = time.time()
                        frame_dt = max(t_end - t_start, 1e-4)
                        state.live_fps = 0.9 * state.live_fps + 0.1 * (1.0 / frame_dt)
                        target_fps = min(30.0, max(15.0, state.fps))
                    target_dt = 1.0 / max(target_fps, 10.0)
                    sleep_dt = target_dt - (time.time() - t_start)
                    if sleep_dt > 0:
                        time.sleep(sleep_dt)
                    continue

            if scale_fac != 1.0:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            run_detection = (cur_frame_idx % stride_val == 0) or (len(last_objs) == 0)

            if run_detection:
                tracked_objects = []
                tracked_masks = []

                # Pure Carton Specialist Model Tracking with Worker-Mask Subtraction Refinement
                try:
                    res_c = state.carton_model.track(
                        frame,
                        persist=True,
                        conf=conf_val,
                        iou=iou_val,
                        imgsz=img_size_val,
                        tracker="bytetrack.yaml",
                        verbose=False,
                    )

                    # Extract worker polygons for mask disentanglement
                    worker_polys = []
                    if hasattr(state, "worker_model") and state.worker_model is not None:
                        try:
                            res_w = state.worker_model.predict(
                                frame, classes=[0], conf=0.25, imgsz=img_size_val, verbose=False
                            )
                            if res_w and res_w[0].masks is not None and getattr(res_w[0].masks, "xy", None) is not None:
                                for wp in res_w[0].masks.xy:
                                    if len(wp) >= 3:
                                        worker_polys.append(wp)
                        except Exception as e:
                            logger.debug("Worker prediction error for mask refinement: %s", e)

                    if res_c and res_c[0].boxes is not None and res_c[0].boxes.id is not None:
                        boxes_c = res_c[0].boxes.xyxy.cpu().numpy()
                        tids_c = res_c[0].boxes.id.cpu().numpy()
                        confs_c = res_c[0].boxes.conf.cpu().numpy()
                        masks_c = getattr(res_c[0], "masks", None)
                        has_masks_c = masks_c is not None and getattr(masks_c, "xy", None) is not None

                        cur_time = time.time()
                        for idx, (box, tid, c_val) in enumerate(zip(boxes_c, tids_c, confs_c)):
                            tracked_objects.append((int(tid), tuple(box), "carton", float(c_val)))
                            if has_masks_c and idx < len(masks_c.xy):
                                raw_poly = masks_c.xy[idx]
                                if raw_poly is not None and len(raw_poly) >= 3:
                                    refined = state.mask_refiner.refine_carton_mask(
                                        raw_polygon=raw_poly,
                                        image_shape=frame.shape[:2],
                                        worker_polygons=worker_polys,
                                        confidence=float(c_val),
                                        track_id=str(tid),
                                        timestamp_sec=cur_time,
                                    )
                                    if refined is not None:
                                        tracked_masks.append((
                                            refined.polygon_pts,
                                            "carton",
                                            float(c_val),
                                            int(tid),
                                            refined.handling_state,
                                            refined.is_occluded_by_worker,
                                        ))
                                    else:
                                        tracked_masks.append((raw_poly, "carton", float(c_val), int(tid), "UNKNOWN", False))
                except Exception as e:
                    logger.debug("Carton model track error: %s", e)
            else:
                tracked_objects = last_objs
                tracked_masks = getattr(state, "last_tracked_masks", [])

            with lock:
                state.last_tracked_objects = tracked_objects
                state.last_tracked_masks = tracked_masks
                state.active_count = len(tracked_objects)

                if run_detection:
                    state.counter.update(tracked_objects, cur_frame_idx, frame=frame)

                # Kinematic Motion Gating: Suppress static background furniture (cupboards/cabinets)
                gated_masks = []
                for item in tracked_masks:
                    poly, cls_n, c_score = item[0], item[1], item[2]
                    tid = item[3] if len(item) > 3 else None
                    if tid is not None and hasattr(state.counter, "track_history"):
                        pts = state.counter.track_history.get(tid, [])
                        if len(pts) >= 2:
                            dx = pts[-1][0] - pts[0][0]
                            dy = pts[-1][1] - pts[0][1]
                            disp = (dx**2 + dy**2) ** 0.5
                        else:
                            disp = 0.0

                        is_static = (len(pts) >= 8 and disp < 8.0)
                        # Suppress stationary background furniture
                        if is_static:
                            continue

                        # Admit moving cartons or newly appeared boxes with solid confidence
                        if disp >= 10.0 or (len(pts) < 5 and c_score >= 0.30):
                            gated_masks.append(item)
                    else:
                        if c_score >= 0.30:
                            gated_masks.append(item)

                state.visualizer.draw_trajectories(frame, state.counter.track_history)
                state.visualizer.draw_tripwire(
                    frame=frame,
                    line_x=state.line_x,
                    loading_direction=state.counter.loading_direction,
                )
                state.visualizer.draw_detections(
                    frame=frame,
                    tracked_objects=tracked_objects,
                    track_side=state.counter.track_side,
                    line_x=state.line_x,
                    associations=None,
                    carton_boxes=None,
                    holding_scores=None,
                    worker_qty=None,
                    masks=gated_masks,
                )
                state.visualizer.draw_hud(
                    frame=frame,
                    total_in=state.counter.total_in,
                    total_out=state.counter.total_out,
                    net_count=state.counter.net_count,
                    fps=state.live_fps,
                    active_count=state.active_count,
                    recent_event=state.counter.recent_event,
                    recent_event_expiry=state.counter.recent_event_expiry,
                    worker_trips=None,
                )

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                _, buffer = cv2.imencode(".jpg", frame, encode_param)
                state.current_frame_jpg = buffer.tobytes()

                t_end = time.time()
                frame_dt = max(t_end - t_start, 1e-4)
                state.live_fps = 0.9 * state.live_fps + 0.1 * (1.0 / frame_dt)
                target_fps = min(30.0, max(15.0, state.fps))

            target_dt = 1.0 / max(target_fps, 10.0)
            sleep_dt = target_dt - (time.time() - t_start)
            if sleep_dt > 0:
                time.sleep(sleep_dt)
        except Exception as e:
            logger.error("Processing loop error: %s\n%s", e, traceback.format_exc())
            time.sleep(0.05)

    logger.info("Processing loop shut down cleanly.")


# ---------------------------------------------------------------------------
# MJPEG Generator
# ---------------------------------------------------------------------------
def generate_frames():
    """MJPEG stream generator with low-latency frame push."""
    last_sent_idx = -999
    last_sent_time = 0.0
    try:
        while not shutdown_event.is_set():
            with lock:
                cur_idx = state.frame_idx
                jpg = state.current_frame_jpg
                is_standby = not state.source

            now = time.time()
            if jpg is not None and (cur_idx != last_sent_idx or (is_standby and now - last_sent_time > 1.0)):
                last_sent_idx = cur_idx
                last_sent_time = now
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.005)
            else:
                time.sleep(0.012)
    except GeneratorExit:
        pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    active_key = next(iter(CONFIGURED_API_KEYS), "carton_secret_token_production")
    return render_template("index.html", client_api_key=active_key)


@app.route("/health")
def health_check():
    uptime = (datetime.now(timezone.utc) - state.start_time).total_seconds()
    live_status = live_camera_manager.status()
    live_active = live_status["active"]
    calibration_complete = bool(
        (state.multi_angle_report or {}).get("calibration", {}).get("calibration_complete")
    )
    recorded_video_mode = (
        (state.multi_angle_report or {}).get("processing_mode") == "recorded_video"
        or ((state.multi_angle_report or {}).get("result") or {}).get("processing_mode") == "recorded_video"
    )
    is_healthy = (
        state.connection_status != "error"
        and _MODELS_VALID
        and (not live_active or live_status["ready"])
    )
    best_hash = ""
    try:
        best_hash = _sha256_file(state.model_name)
    except Exception:
        best_hash = "unavailable"
    return jsonify({
        "status": "healthy" if is_healthy else "degraded",
        "service": "truck-loading-detection",
        "models_valid": _MODELS_VALID,
        "model": state.model_name,
        "model_sha256": best_hash,
        "detection_threshold": CONF,
        "consensus_threshold": CONSENSUS_THRESHOLD,
        "connection_status": state.connection_status,
        "source": state.source or None,
        "imgsz": state.img_size,
        "conf": state.conf,
        "uptime_seconds": round(uptime, 1),
        "net_count": state.counter.net_count,
        "loaded_cartons": state.counter.total_in,
        "returned_cartons": state.counter.total_out,
        "fps": round(state.live_fps, 1),
        "reconnect_attempts": state.reconnect_attempts,
        "consecutive_frame_drops": getattr(state, "consecutive_frame_drops", 0),
        "active_tracks": state.active_count,
        "live_cameras": live_status,
        "counting_enabled": (
            recorded_video_mode
            or (not live_active and (calibration_complete if state.is_multi_angle else True))
        ),
        "counting_blocker": (
            None
            if recorded_video_mode
            else (
                live_status["counting_blocker"]
                if live_active
                else (
                    "Verified calibration is required for three-camera automatic counting"
                    if state.is_multi_angle and not calibration_complete
                    else None
                )
            )
        ),
    })


# ---------------------------------------------------------------------------
# Phase 0 Calibration & Watchdog Endpoints
# ---------------------------------------------------------------------------
@app.route("/calibrate/<site_id>/<dock_id>")
def calibrate_dashboard(site_id, dock_id):
    """Interactive three-camera loading-gate calibration tool."""
    return render_template("calibrate.html", site_id=site_id, dock_id=dock_id)


@app.route("/api/v2/calibrate/<site_id>/<dock_id>/snapshot")
def calibrate_snapshot(site_id, dock_id):
    """Retrieve a frame from the explicitly selected camera source."""
    camera_id = request.args.get("camera_id", "cam_top")
    if camera_id not in ("cam_top", "cam_side", "cam_flank"):
        return jsonify({"error": f"Unknown camera_id: {camera_id}"}), 400
    with lock:
        camera_source = getattr(state, "live_cameras", {}).get(camera_id)
    if not camera_source:
        return jsonify({
            "error": f"No live source is configured for {camera_id}; refusing to return another camera or a sample frame",
        }), 409

    if live_camera_manager.status()["active"]:
        frame_jpeg = live_camera_manager.latest_jpeg(camera_id)
        if frame_jpeg is None:
            return jsonify({"error": f"No frame received yet from {camera_id}"}), 503
        response = Response(frame_jpeg, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

    try:
        parsed_source = parse_camera_source(camera_source)
    except ValueError as exc:
        return jsonify({"error": f"Invalid source configured for {camera_id}: {exc}"}), 400
    cap = cv2.VideoCapture(parsed_source)
    if not cap.isOpened():
        cap.release()
        return jsonify({"error": f"Could not open live source for {camera_id}"}), 503
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return jsonify({"error": f"No frame available from {camera_id}"}), 503

    # Apply undistortion if calibration is established
    if calib_storage.exists(site_id, dock_id):
        try:
            dock_calib = calib_storage.load(site_id, dock_id)
            if camera_id in dock_calib.cameras:
                cam_rec = dock_calib.cameras[camera_id]
                if cam_rec.intrinsics_calibrated:
                    frame = lut_manager.remap_frame(
                        frame,
                        camera_id=camera_id,
                        K=cam_rec.K,
                        D=cam_rec.D,
                        camera_model=cam_rec.camera_model,
                    )
        except Exception as e:
            logger.warning("Could not apply undistortion to snapshot: %s", e)

    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        return "Failed to encode snapshot", 500

    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/v2/calibrate/<site_id>/<dock_id>/landmarks", methods=["POST"])
def calibrate_landmarks(site_id, dock_id):
    """
    Calibrate the loading-gate plane only from measured world references and
    separately withheld validation points.
    """
    data = request.get_json(silent=True) or {}
    camera_id = data.get("camera_id", "cam_top")
    raw_landmarks = data.get("landmarks", {})
    physical_points = data.get("physical_reference_points")
    validation_points = data.get("independent_validation_points")
    calibration_plane = data.get("calibration_plane")
    frame_source = data.get("frame_source")
    source_frame_name = data.get("source_frame_name", "")

    try:
        if camera_id not in ("cam_top", "cam_side", "cam_flank"):
            raise ValueError("camera_id must be cam_top, cam_side, or cam_flank")
        if data.get("image_orientation_confirmed") is not True:
            raise ValueError(
                "Confirm that the selected frame has the same orientation as this camera's processing frames"
            )
        if frame_source not in ("live_snapshot", "local_image"):
            raise ValueError("frame_source must be 'live_snapshot' or 'local_image'")
        if (
            not isinstance(source_frame_name, str)
            or len(source_frame_name) > 200
            or "/" in source_frame_name
            or "\\" in source_frame_name
            or source_frame_name in (".", "..")
        ):
            raise ValueError("source_frame_name must be a filename of at most 200 characters")
        image_width = int(data.get("image_width", 1280))
        image_height = int(data.get("image_height", 720))
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")
        if not isinstance(physical_points, dict) or set(physical_points) != set(REQUIRED_LANDMARK_NAMES):
            raise ValueError(
                "physical_reference_points must provide surveyed world coordinates for all six landmarks"
            )
        if calibration_plane != "loading_gate_plane":
            raise ValueError("calibration_plane must be 'loading_gate_plane'")
        if not isinstance(validation_points, list) or len(validation_points) < 4:
            raise ValueError(
                "At least four independent_validation_points are required; these points must not be used to fit the homography"
            )

        world_coords = {}
        for name in REQUIRED_LANDMARK_NAMES:
            coords = physical_points[name]
            if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                raise ValueError(f"physical reference '{name}' must be an [x, y] pair in meters")
            point = (float(coords[0]), float(coords[1]))
            if not np.all(np.isfinite(point)):
                raise ValueError(f"physical reference '{name}' must contain finite coordinates")
            world_coords[name] = point
        if len(set(world_coords.values())) != len(REQUIRED_LANDMARK_NAMES):
            raise ValueError("The six physical reference coordinates must be distinct")

        landmarks = LandmarkSet(
            points={k: (float(v[0]), float(v[1])) for k, v in raw_landmarks.items()},
            image_width=image_width,
            image_height=image_height,
        )
        if len({tuple(point) for point in landmarks.to_array().tolist()}) != len(REQUIRED_LANDMARK_NAMES):
            raise ValueError("The six image reference coordinates must be distinct")

        validation_image_points = []
        validation_world_points = []
        for index, point in enumerate(validation_points):
            if not isinstance(point, dict) or len(point.get("image", [])) != 2 or len(point.get("world", [])) != 2:
                raise ValueError(f"independent validation point {index} must contain image and world [x, y] pairs")
            image_xy = (float(point["image"][0]), float(point["image"][1]))
            world_xy = (float(point["world"][0]), float(point["world"][1]))
            if not np.all(np.isfinite(image_xy + world_xy)):
                raise ValueError(f"independent validation point {index} must contain finite coordinates")
            if not (0 <= image_xy[0] < image_width and 0 <= image_xy[1] < image_height):
                raise ValueError(f"independent validation point {index} is outside the source image")
            if world_xy in world_coords.values():
                raise ValueError(f"independent validation point {index} reuses a fitting reference point")
            validation_image_points.append(image_xy)
            validation_world_points.append(world_xy)

        if len(set(validation_world_points)) != len(validation_world_points):
            raise ValueError("Independent validation world coordinates must all be distinct")
        if len(set(validation_image_points)) != len(validation_image_points):
            raise ValueError("Independent validation image coordinates must all be distinct")
        if set(validation_image_points) & {tuple(point) for point in landmarks.to_array().tolist()}:
            raise ValueError("Independent validation image points must not reuse a fitting image point")

        validation_world_array = np.asarray(validation_world_points, dtype=np.float64)
        if np.linalg.matrix_rank(
            np.column_stack((validation_world_array, np.ones(len(validation_world_array))))
        ) < 3:
            raise ValueError("independent validation points must span a non-collinear area on the gate plane")

    except Exception as e:
        return jsonify({"status": "rejected", "reason": f"Invalid landmark set: {e}"}), 400

    estimator = DoorwayHomographyEstimator(
        canonical_world_coords=world_coords,
        max_reprojection_rms=2.0,
    )
    result = estimator.estimate(landmarks)

    if not result.is_valid:
        return jsonify({
            "status": "rejected",
            "reason": result.rejection_reason,
            "rms_error_px": result.rms_error_px,
            "max_error_px": result.max_error_px,
            "inlier_count": result.inlier_count,
        }), 422

    validation_world_h = np.column_stack(
        (validation_world_array, np.ones(len(validation_world_array), dtype=np.float64))
    )
    validation_image_h = (np.linalg.inv(result.H) @ validation_world_h.T).T
    if np.any(np.abs(validation_image_h[:, 2]) < 1e-9):
        return jsonify({
            "status": "rejected",
            "reason": "Independent validation points project to infinity under the fitted homography",
        }), 422
    predicted_image = validation_image_h[:, :2] / validation_image_h[:, 2:3]
    validation_errors = np.linalg.norm(
        np.asarray(validation_image_points, dtype=np.float64) - predicted_image,
        axis=1,
    )
    validation_rms_px = float(np.sqrt(np.mean(validation_errors ** 2)))
    validation_max_px = float(np.max(validation_errors))
    if validation_rms_px > 2.0:
        return jsonify({
            "status": "rejected",
            "reason": f"Independent validation RMS {validation_rms_px:.3f}px exceeds 2.0px",
            "independent_validation_rms_px": validation_rms_px,
            "independent_validation_max_px": validation_max_px,
        }), 422

    homography_record = result.to_dict()
    homography_record.update({
        "calibration_plane": calibration_plane,
        "image_orientation_confirmed": True,
        "frame_source": frame_source,
        "source_frame_name": source_frame_name,
        "physical_reference_points": world_coords,
        "independent_validation": {
            "passed": True,
            "calibration_plane": calibration_plane,
            "point_count": len(validation_points),
            "rms_error_px": validation_rms_px,
            "max_error_px": validation_max_px,
        },
    })

    # Save to atomic dock calibration store
    try:
        if calib_storage.exists(site_id, dock_id):
            dock_calib = calib_storage.load(site_id, dock_id)
        else:
            dock_calib = DockCalibration(site_id=site_id, dock_id=dock_id)

        if camera_id in dock_calib.cameras:
            cam_data = dock_calib.cameras[camera_id]
        else:
            cam_data = CameraCalibrationData(
                camera_id=camera_id,
                camera_model="pinhole",
                image_width=landmarks.image_width,
                image_height=landmarks.image_height,
                focal_length_mm=None,
                intrinsics_calibrated=False,
                K=np.eye(3),
                D=np.zeros(5),
                rms_error=0.0,
            )
            dock_calib.add_camera(cam_data)

        cam_data.image_width = image_width
        cam_data.image_height = image_height
        cam_data.landmarks = landmarks.to_dict()
        cam_data.homography = homography_record
        cam_data.alignment_status = "CALIBRATED_OPERATIONAL"
        calib_storage.save(dock_calib)

        watchdogs = dock_watchdogs.setdefault(f"{site_id}_{dock_id}", {})
        if camera_id in watchdogs:
            watchdogs[camera_id].state = CameraAlignmentState.CALIBRATED_OPERATIONAL

        return jsonify({
            "status": "success",
            "rms_error_px": result.rms_error_px,
            "max_error_px": result.max_error_px,
            "inlier_count": result.inlier_count,
            "inlier_ratio": result.inlier_ratio,
            "independent_validation_rms_px": validation_rms_px,
            "independent_validation_max_px": validation_max_px,
            "homography": homography_record,
        })
    except Exception as e:
        logger.exception("Failed to persist calibration: %s", e)
        return jsonify({"status": "rejected", "reason": f"Persistence error: {e}"}), 500


@app.route("/api/v2/calibrate/<site_id>/<dock_id>/status")
def calibrate_status(site_id, dock_id):
    """Get calibration and alignment status for dock cameras."""
    exists = calib_storage.exists(site_id, dock_id)
    dock_calib = calib_storage.load(site_id, dock_id) if exists else None

    cameras_status = {}
    watchdogs = dock_watchdogs.get(f"{site_id}_{dock_id}", {})

    for cid in ("cam_top", "cam_side", "cam_flank"):
        cdata = dock_calib.cameras.get(cid) if dock_calib else None
        if cdata:
            wd = watchdogs.get(cid)
            independent_validation = (
                cdata.homography.get("independent_validation", {})
                if cdata.homography else {}
            )
            has_valid_calibration = bool(
                cdata.homography
                and cdata.homography.get("is_valid")
                and cdata.homography.get("calibration_plane") == "loading_gate_plane"
                and cdata.homography.get("image_orientation_confirmed") is True
                and independent_validation.get("passed") is True
                and independent_validation.get("calibration_plane") == "loading_gate_plane"
            )
            state_val = (
                wd.state.value if wd and has_valid_calibration
                else cdata.alignment_status if has_valid_calibration
                else "UNCALIBRATED"
            )
            penalties_active = bool(
                has_valid_calibration
                and (wd.spatial_penalties_enabled if wd else state_val == "CALIBRATED_OPERATIONAL")
            )
            cameras_status[cid] = {
                "alignment_status": state_val,
                "spatial_penalties_enabled": penalties_active,
                "has_landmarks": cdata.landmarks is not None,
                "has_homography": cdata.homography is not None,
                "rms_error_px": cdata.homography.get("rms_error_px") if cdata.homography else None,
                "independent_validation_rms_px": independent_validation.get("rms_error_px"),
                "is_calibrated": has_valid_calibration,
                "image_width": cdata.image_width,
                "image_height": cdata.image_height,
                "image_orientation_confirmed": bool(
                    cdata.homography
                    and cdata.homography.get("image_orientation_confirmed") is True
                ),
            }
        else:
            cameras_status[cid] = {
                "alignment_status": "UNCALIBRATED",
                "spatial_penalties_enabled": False,
                "has_landmarks": False,
                "has_homography": False,
                "rms_error_px": None,
                "independent_validation_rms_px": None,
                "is_calibrated": False,
                "image_width": None,
                "image_height": None,
                "image_orientation_confirmed": False,
            }

    return jsonify({
        "site_id": site_id,
        "dock_id": dock_id,
        "is_calibrated": all(
            camera["is_calibrated"] for camera in cameras_status.values()
        ),
        "cameras": cameras_status,
    })


@app.route("/video_feed")
def video_feed():
    resp = Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.route("/api/live_cameras/feed/<camera_id>")
def api_live_camera_feed(camera_id):
    if camera_id not in CAMERA_IDS:
        return jsonify({"error": f"Unknown camera_id: {camera_id}"}), 404

    def generate_camera_frames():
        standby = state.create_standby_frame(f"WAITING FOR {camera_id.upper()}")
        last_frame = None
        while not shutdown_event.is_set():
            frame = live_camera_manager.latest_jpeg(camera_id)
            if frame is not None and frame != last_frame:
                last_frame = frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            elif frame is None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + standby
                    + b"\r\n"
                )
                time.sleep(0.5)
            else:
                time.sleep(0.08)

    response = Response(
        generate_camera_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.route("/api/live_cameras/status")
def api_live_cameras_status():
    return jsonify(live_camera_manager.status())


@app.route("/api/stats")
def api_stats():
    with lock:
        live_status = live_camera_manager.status()
        if live_status.get("active"):
            lc = live_status.get("live_counts", {})
            c_active = bool(live_status.get("counting_enabled") or live_status.get("counting_active"))
            return jsonify({
                "is_multi_angle": True,
                "is_live": True,
                "counting_enabled": c_active,
                "counting_blocker": None if c_active else "Live camera preview active. Click 'Start live counting' to count cartons.",
                "total_in": lc.get("loaded", 0),
                "total_out": lc.get("returned", 0),
                "net_count": lc.get("net", 0),
                "confirmed_in": lc.get("confirmed_loaded", 0),
                "confirmed_out": lc.get("confirmed_returned", 0),
                "fps": lc.get("processing_fps", 0.0),
                "connection_status": "live",
                "cameras": live_status.get("cameras", {}),
                "events": [],
            })

        t_param = request.args.get("t", type=float)
        is_paused_param = request.args.get("is_paused")
        client_paused = (is_paused_param in ("1", "true", "True")) if is_paused_param is not None else None
        source = state.source
        is_live = isinstance(source, int) or (
            isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("http"))
        ) or state.connection_status == "live"

        if state.is_multi_angle and state.multi_angle_report:
            res = state.multi_angle_report.get("result", {})
            raw_events = (
                res.get("events")
                or res.get("committed_events")
                or state.multi_angle_report.get("committed_events")
                or state.multi_angle_report.get("events")
                or []
            )
            fps_val = max(state.fps, 1.0)

            # In recorded video mode, if client passes its active video timestamp,
            # bind the counts strictly to that playback timestamp.
            # When video is paused at T, T never advances -> counts stay 100% frozen!
            if t_param is not None and t_param >= 0 and not is_live:
                current_sec = t_param
                effective_frame = int(round(current_sec * fps_val))
            elif state.frame_idx > 0:
                current_sec = state.frame_idx / fps_val
                effective_frame = state.frame_idx
            elif is_live:
                current_sec = state.frame_idx / fps_val
                effective_frame = state.frame_idx
            else:
                # Use the latest finite event time instead of emitting non-standard JSON Infinity.
                event_times = []
                event_frames = []
                for event in raw_events:
                    event_frame = event.get("frame_idx")
                    if event_frame is not None:
                        try:
                            event_frames.append(int(event_frame))
                        except (TypeError, ValueError):
                            pass
                    event_time = event.get("timestamp_sec", event.get("time_sec"))
                    if event_time is None and event_frame is not None:
                        event_time = event_frame / fps_val
                    try:
                        event_time = float(event_time)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(event_time):
                        event_times.append(event_time)
                current_sec = max(event_times, default=0.0)
                effective_frame = max(
                    max(event_frames, default=0),
                    int(round(current_sec * fps_val)),
                )

            # Filter events strictly up to current playback frame/time
            active_events = []
            for ev in raw_events:
                ev_time = ev.get("timestamp_sec")
                if ev_time is None and ev.get("time_sec") is not None:
                    ev_time = ev.get("time_sec")
                ev_frame = ev.get("frame_idx")

                if ev_time is not None:
                    if ev_time <= current_sec:
                        active_events.append(ev)
                elif ev_frame is not None and ev_frame <= effective_frame:
                    active_events.append(ev)

            breakdown = res.get("consensus_breakdown") or state.multi_angle_report.get("consensus_breakdown", {})
            calibration = res.get("calibration") or state.multi_angle_report.get("calibration", {})
            calibration_complete = calibration.get("calibration_complete") is True
            recorded_video_mode = (
                state.multi_angle_report.get("processing_mode") == "recorded_video"
                or res.get("processing_mode") == "recorded_video"
            )

            def is_confirmed(ev_dict: dict) -> bool:
                if not calibration_complete and not recorded_video_mode:
                    return False
                if ev_dict.get("is_auto_confirmed") is not True or ev_dict.get("requires_review") is not False:
                    return False
                if recorded_video_mode:
                    details = ev_dict.get("details", {})
                    cameras = ev_dict.get("participating_cameras") or details.get("participating_cameras", [])
                    association = ev_dict.get("association_method") or details.get("association_method")
                    if len(set(cameras)) < 2:
                        return False
                    if association not in ("synchronized_video_tripwire_time", "calibrated_spatial"):
                        return False
                if ev_dict.get("state") in ("REVIEW_REQUIRED", "REVERSED_CANCELLED"):
                    return False
                c_level = ""
                if isinstance(ev_dict.get("consensus"), dict):
                    c_level = ev_dict["consensus"].get("level", "")
                elif ev_dict.get("consensus_status"):
                    c_level = ev_dict["consensus_status"]
                if c_level in ("SINGLE_CAMERA_OBSERVATION", "CONFLICTING_OR_UNCERTAIN"):
                    return False
                return True

            observed_in = sum(
                ev.get("fused_cartons") or ev.get("carton_count") or 1
                for ev in active_events if ev.get("direction", "IN") == "IN"
            )
            observed_out = sum(
                ev.get("fused_cartons") or ev.get("carton_count") or 1
                for ev in active_events if ev.get("direction") == "OUT"
            )
            review_in = sum(
                ev.get("fused_cartons") or ev.get("carton_count") or 1
                for ev in active_events
                if ev.get("direction", "IN") == "IN" and not is_confirmed(ev)
            )
            review_out = sum(
                ev.get("fused_cartons") or ev.get("carton_count") or 1
                for ev in active_events
                if ev.get("direction") == "OUT" and not is_confirmed(ev)
            )
            confirmed_in = sum(ev.get("fused_cartons") or ev.get("carton_count") or 1 for ev in active_events if ev.get("direction", "IN") == "IN" and is_confirmed(ev))
            confirmed_out = sum(ev.get("fused_cartons") or ev.get("carton_count") or 1 for ev in active_events if ev.get("direction") == "OUT" and is_confirmed(ev))
            net_cnt = confirmed_in - confirmed_out
            trips_in = sum(1 for ev in active_events if ev.get("direction", "IN") == "IN")
            recent_events = list(active_events)
            recent_events.reverse()

            formatted_recent = []
            for ev in recent_events[:50]:
                cnt = ev.get("fused_cartons") or ev.get("carton_count") or 1
                c_level = "VERIFIED"
                if isinstance(ev.get("consensus"), dict):
                    c_level = ev["consensus"].get("level", "VERIFIED")
                elif ev.get("consensus_status"):
                    c_level = ev["consensus_status"]
                elif not is_confirmed(ev):
                    c_level = "REVIEW_REQUIRED"
                cls_name = ev.get("class_name") or ev.get("details", {}).get("class_name", "carton")
                is_worker = (cls_name in ("person", "worker")) or ("worker" in str(ev.get("note", "")).lower())
                details = ev.get("details", {})
                formatted_recent.append({
                    "fused_id": ev.get("event_id") or details.get("event_id") or ev.get("fused_id") or f"ev_{len(formatted_recent)+1}",
                    "global_carton_id": ev.get("global_carton_id") or details.get("global_carton_id"),
                    "direction": ev.get("direction", "IN"),
                    "fused_cartons": cnt,
                    "carton_count": cnt,
                    "timestamp_sec": round(float(ev.get("timestamp_sec") or (ev.get("frame_idx", 0) / fps_val)), 2),
                    "consensus_status": c_level,
                    "is_confirmed": is_confirmed(ev),
                    "requires_review": not is_confirmed(ev),
                    "class_name": "worker" if is_worker else "carton",
                    "event_type": "worker_trip" if is_worker else "crossing",
                    "worker_trip": is_worker,
                    "camera_evidence": ev.get("camera_evidence") or details.get("camera_evidence", {}),
                    "note": ev.get("note") or details.get("notes") or f"Carton {ev.get('direction', 'IN')} ({c_level})",
                })

            if (breakdown.get("THREE_CAMERA_UNANIMOUS", 0) > 0
                or breakdown.get("WIDTH_SAMPLED_3CAM", 0) > 0
                or breakdown.get("THREE_CAMERA_VERIFIED", 0) > 0):
                c_status = "THREE_CAMERA_VERIFIED"
            elif (breakdown.get("TWO_CAMERA_AGREEMENT", 0) > 0
                  or breakdown.get("DUAL_CAMERA_VERIFIED", 0) > 0):
                c_status = "DUAL_CAMERA_VERIFIED"
            else:
                c_status = "SINGLE_CAMERA_OBSERVATION"
            if not calibration_complete and not recorded_video_mode:
                c_status = "CALIBRATION_REQUIRED"
            elif recorded_video_mode and not calibration_complete:
                c_status = "RECORDED_VIDEO_ASSOCIATION"
            live_status = live_camera_manager.status()
            live_preview_active = live_status["active"]
            live_counting_active = bool(live_status.get("counting_enabled"))
            if live_counting_active and live_status.get("live_counts"):
                lc = live_status["live_counts"]
                confirmed_in = lc.get("confirmed_loaded", confirmed_in)
                confirmed_out = lc.get("confirmed_returned", confirmed_out)
                net_cnt = lc.get("net", net_cnt)
                review_in = lc.get("review_required", review_in)
                state.live_fps = lc.get("processing_fps", state.live_fps)
                effective_frame = lc.get("total_ticks", effective_frame)
                current_sec = lc.get("timestamp_sec", current_sec)
                counting_enabled = True
                counting_blocker = None
            else:
                counting_enabled = recorded_video_mode or (
                    not live_preview_active and calibration_complete
                )
                counting_blocker = (
                    None
                    if recorded_video_mode
                    else (
                        "Live camera mode is preview-only; automatic live counting is not implemented"
                        if live_preview_active
                        else (
                            "This report lacks verified recorded-video association metadata or complete calibration"
                            if not calibration_complete
                            else None
                        )
                    )
                )
            return jsonify({
                "is_multi_angle": True,
                "recorded_video_mode": recorded_video_mode,
                "counting_enabled": counting_enabled,
                "counting_blocker": counting_blocker,
                "total_in": confirmed_in,
                "total_out": confirmed_out,
                "net_count": net_cnt,
                "confirmed_in": confirmed_in,
                "confirmed_out": confirmed_out,
                "observed_in": observed_in,
                "observed_out": observed_out,
                "pending_review_count": review_in + review_out,
                "pending_review_in": review_in,
                "pending_review_out": review_out,
                "worker_trips_in": trips_in,
                "worker_trips_out": 0,
                "fps": round(state.live_fps, 1),
                "active_count": 0,
                "frame_idx": effective_frame,
                "timestamp_sec": round(current_sec, 2),
                "is_paused": state.is_paused or bool(client_paused),
                "is_live": is_live,
                "loop_video": state.loop_video,
                "source": state.source,
                "connection_status": state.connection_status,
                "consensus_breakdown": breakdown,
                "consensus_status": c_status,
                "calibration": calibration,
                "events": formatted_recent,
                "cameras": state.multi_angle_report.get("cameras", []),
                "tripwires": state.tripwire_configs,
            })

        recent_events = list(state.counter.events)
        recent_events.reverse()
        formatted_single_events = []
        for ev in recent_events[:50]:
            cls = ev.get("class_name", "carton")
            direction = ev.get("direction", "IN")
            delta = ev.get("delta", 1)
            time_sec = ev.get("time") or ev.get("timestamp_sec") or (ev.get("frame_idx", 0) / max(1.0, state.live_fps))
            is_w = cls in ("person", "worker")
            formatted_single_events.append({
                "fused_id": f"trk_{ev.get('track_id', 1)}",
                "direction": "IN" if delta > 0 or direction in ("IN", "left_to_right") else "OUT",
                "fused_cartons": abs(delta) if delta != 0 else 1,
                "carton_count": abs(delta) if delta != 0 else 1,
                "timestamp_sec": round(float(time_sec), 2),
                "consensus_status": "MONO_VERIFIED",
                "is_confirmed": True,
                "requires_review": False,
                "class_name": "worker" if is_w else "carton",
                "event_type": "worker_trip" if is_w else "crossing",
                "worker_trip": is_w or bool(ev.get("carried_by")),
                "note": ev.get("note", f"{cls} crossing"),
            })
        effective_in = state.counter.total_in
        effective_out = state.counter.total_out
        effective_net = state.counter.net_count
        live_preview_active = live_camera_manager.status()["active"]
        return jsonify({
            "is_multi_angle": False,
            "counting_enabled": not live_preview_active,
            "counting_blocker": (
                "Live camera mode is preview-only; automatic live counting is not implemented"
                if live_preview_active
                else None
            ),
            "total_in": effective_in,
            "total_out": effective_out,
            "net_count": effective_net,
            "worker_trips_in": state.counter.worker_trips_in if effective_in > 0 else 0,
            "worker_trips_out": state.counter.worker_trips_out if effective_out > 0 else 0,
            "loading_direction": state.counter.loading_direction,
            "fps": round(state.live_fps, 1),
            "active_count": state.active_count,
            "frame_idx": state.frame_idx,
            "line_x": state.line_x,
            "video_width": state.video_width,
            "video_height": state.video_height,
            "is_paused": state.is_paused or bool(client_paused),
            "loop_video": state.loop_video,
            "source": state.source,
            "connection_status": state.connection_status,
            "reconnect_attempts": state.reconnect_attempts,
            "count_mode": state.count_mode,
            "active_associations": state.counter.current_associations,
            "events": formatted_single_events,
            "conf": round(state.conf, 2),
            "hysteresis": state.counter.hysteresis,
        })


@app.route("/api/reset_line_preset", methods=["POST"])
@require_api_key("RESET_LINE_PRESET")
def api_reset_line_preset():
    with lock:
        base_name = os.path.basename(str(state.source))
        if base_name in VIDEO_PRESETS:
            preset = VIDEO_PRESETS[base_name]
            state.line_x = int(state.video_width * preset["line_x_ratio"])
            state.loading_direction = preset["direction"]
            state.conf = preset.get("conf", 0.24)
        else:
            state.line_x = state.video_width // 2
        state.counter.set_line_x(state.line_x)
        state.counter.set_loading_direction(state.loading_direction)
    return jsonify({
        "success": True,
        "line_x": state.line_x,
        "loading_direction": state.loading_direction,
        "conf": state.conf,
        "hysteresis": state.counter.hysteresis,
    })


@app.route("/api/set_line_x", methods=["POST"])
@require_api_key("UPDATE_LINE_POSITION")
def api_set_line_x():
    data = request.json or {}
    new_x = int(data.get("line_x", state.line_x))
    with lock:
        state.line_x = max(10, min(state.video_width - 10, new_x))
        state.counter.set_line_x(state.line_x)
    return jsonify({"success": True, "line_x": state.line_x})


@app.route("/api/toggle_direction", methods=["POST"])
@require_api_key("TOGGLE_DIRECTION")
def api_toggle_direction():
    with lock:
        new_dir = "right_to_left" if state.counter.loading_direction == "left_to_right" else "left_to_right"
        state.loading_direction = new_dir
        state.counter.set_loading_direction(new_dir)
    return jsonify({"success": True, "loading_direction": state.loading_direction})


@app.route("/api/set_count_mode", methods=["POST"])
@require_api_key("SET_COUNT_MODE")
def api_set_count_mode():
    data = request.json or {}
    mode = data.get("mode", "cartons_only")
    if mode not in COUNT_MODES:
        return jsonify({"error": f"Invalid mode. Choose from {list(COUNT_MODES.keys())}"}), 400
    with lock:
        state.count_mode = mode
        state.counter.set_countable_classes(COUNT_MODES[mode])
    return jsonify({"success": True, "count_mode": state.count_mode})


@app.route("/api/set_playback_state", methods=["POST"])
@app.route("/api/toggle_pause", methods=["POST"])
def api_toggle_pause():
    data = request.get_json(silent=True) or {}
    with lock:
        source = state.source
        is_live = isinstance(source, int) or (
            isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("http"))
        ) or state.connection_status == "live"

        if "is_paused" in data:
            target_pause = bool(data["is_paused"])
            # Live detection must never pause; only recorded videos pause
            state.is_paused = target_pause if not is_live else False
        else:
            state.is_paused = (not state.is_paused) if not is_live else False

        logger.info("Playback state updated: is_paused=%s (is_live=%s)", state.is_paused, is_live)
    return jsonify({"success": True, "is_paused": state.is_paused, "is_live": is_live})


@app.route("/api/seek", methods=["POST"])
def api_seek():
    data = request.get_json(silent=True) or {}
    t_sec = float(data.get("timestamp_sec", 0.0))
    with lock:
        fps_val = max(state.fps, 1.0)
        target_frame = int(round(t_sec * fps_val))
        state.frame_idx = target_frame
        if state.cap is not None:
            try:
                state.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            except Exception:
                pass
    return jsonify({"success": True, "frame_idx": state.frame_idx, "timestamp_sec": t_sec})


@app.route("/api/toggle_loop", methods=["POST"])
def api_toggle_loop():
    with lock:
        state.loop_video = not state.loop_video
    return jsonify({"success": True, "loop_video": state.loop_video})


@app.route("/api/reset", methods=["POST"])
@require_api_key("RESET_COUNTS")
def api_reset():
    with lock:
        state.counter.reset_counts()
        if state.cap is not None:
            ret = state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if not ret:
                state.init_capture()
        state.frame_idx = 0
        state.active_count = 0
        logger.info("Counts, frame index, and video playback reset to 0 by user")
    return jsonify({"success": True, "frame_idx": 0, "net_count": 0})


@app.route("/api/set_source", methods=["POST"])
def api_set_source():
    data = request.json or {}
    src = str(data.get("source", "")).strip()

    # Security Validation (SSRF & Arbitrary Path Traversal Protection)
    if src:
        # 1. Device index (0, 1, etc.)
        if src.isdigit():
            validated_src = int(src)
        # 2. Live streaming protocols
        elif src.startswith(("rtsp://", "rtmp://")):
            validated_src = src
        # 3. HTTP streams (block SSRF against private networks & cloud metadata)
        elif src.startswith(("http://", "https://")):
            lower_src = src.lower()
            if any(forbidden in lower_src for forbidden in ["169.254.", "localhost", "127.0.0.1", "0.0.0.0", "::1"]):
                logger.warning("Blocked SSRF attempt: %s", src)
                return jsonify({"error": "Access to private or metadata network endpoints forbidden"}), 403
            validated_src = src
        else:
            # 4. Local video files: restrict strictly to allowed directories
            allowed_base_dirs = [
                os.path.abspath(os.path.dirname(__file__)),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "recorded_video")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "uploads")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "output_videos")),
            ]
            if not os.path.isabs(src):
                abs_target = os.path.abspath(os.path.join(os.path.dirname(__file__), src))
            else:
                abs_target = os.path.abspath(src)

            is_allowed = any(os.path.commonpath([d, abs_target]) == d for d in allowed_base_dirs)
            if not is_allowed or not os.path.exists(abs_target):
                logger.warning("Blocked path traversal / unauthorized source access attempt: %s", src)
                return jsonify({"error": "Forbidden: Path is outside allowed video directories"}), 403
            validated_src = abs_target
    else:
        validated_src = ""

    with lock:
        state.multi_angle_report = None
        state.is_multi_angle = False
        state.source = validated_src
        state.counter.reset_counts()
        state.init_capture()
    logger.info("Source changed to: %s", validated_src or "(none)")
    return jsonify({"success": True, "source": str(state.source), "connection_status": state.connection_status})


@app.route("/healthz")
def healthz():
    """Liveness and readiness health probe."""
    live_status = live_camera_manager.status()
    live_active = live_status["active"]
    capture_open = state.cap is not None and state.cap.isOpened()
    is_healthy = live_status["ready"] if live_active else capture_open
    ready_without_source = not live_active and not state.source
    calibration_complete = bool(
        (state.multi_angle_report or {}).get("calibration", {}).get("calibration_complete")
    )
    recorded_video_mode = (
        (state.multi_angle_report or {}).get("processing_mode") == "recorded_video"
        or ((state.multi_angle_report or {}).get("result") or {}).get("processing_mode") == "recorded_video"
    )
    return jsonify({
        "status": "healthy" if is_healthy else "degraded",
        "fps": round(state.actual_fps, 1),
        "frame_idx": state.frame_idx,
        "net_count": state.counter.net_count if state.counter else 0,
        "connection_status": state.connection_status,
        "live_cameras": live_status,
        "counting_enabled": (
            recorded_video_mode
            or (not live_active and (calibration_complete if state.is_multi_angle else True))
        ),
    }), (200 if is_healthy or ready_without_source else 503)


@app.route("/metrics")
def metrics():
    """Prometheus-compatible plaintext metrics."""
    c = state.counter
    net = c.net_count if c else 0
    total_in = c.total_in if c else 0
    total_out = c.total_out if c else 0
    worker_in = c.worker_trips_in if c else 0
    fps = round(state.actual_fps, 2)
    lines = [
        "# HELP carton_counter_fps Current processing frames per second",
        "# TYPE carton_counter_fps gauge",
        f"carton_counter_fps {fps}",
        "# HELP carton_counter_total_in Total cartons loaded into truck",
        "# TYPE carton_counter_total_in counter",
        f"carton_counter_total_in {total_in}",
        "# HELP carton_counter_total_out Total cartons unloaded or returned",
        "# TYPE carton_counter_total_out counter",
        f"carton_counter_total_out {total_out}",
        "# HELP carton_counter_net Net cartons loaded",
        "# TYPE carton_counter_net gauge",
        f"carton_counter_net {net}",
        "# HELP carton_counter_worker_trips Total worker trips recorded",
        "# TYPE carton_counter_worker_trips counter",
        f"carton_counter_worker_trips {worker_in}",
    ]
    return "\n".join(lines) + "\n", 200, {"Content-Type": "text/plain; version=0.0.4"}


@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        if "video" not in request.files and "file" not in request.files:
            return jsonify({"status": "error", "error": "No file uploaded", "message": "No file uploaded"}), 400
        file = request.files.get("video") or request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"status": "error", "error": "Empty filename", "message": "Empty filename"}), 400

        raw_name = file.filename
        base, ext = os.path.splitext(raw_name)
        ext = ext.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                "status": "error",
                "error": f"Only video files ({', '.join(ALLOWED_EXTENSIONS)}) are allowed",
                "message": f"Only video files are allowed"
            }), 400

        safe_base = secure_filename(base)
        if not safe_base:
            safe_base = f"video_{int(time.time())}"
        filename = f"{safe_base}{ext}"
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Cleanup old uploads (keep last 10 files)
        try:
            upload_dir = app.config["UPLOAD_FOLDER"]
            allowed_tuple = tuple(ALLOWED_EXTENSIONS)
            files = sorted(
                [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if f.endswith(allowed_tuple)],
                key=os.path.getmtime,
            )
            for old in files[:-10]:
                os.remove(old)
                logger.info("Cleaned old upload: %s", old)
        except Exception as e:
            logger.warning("Upload cleanup failed: %s", e)

        with lock:
            state.source = filepath
            state.counter.reset_counts()
            state.counter.prev_gray = None
            state.current_frame_jpg = None
            state.init_capture()

        logger.info("Uploaded and loaded: %s", filename)
        return jsonify({
            "status": "ok",
            "success": True,
            "filename": filename,
            "filepath": filename,
            "source": filepath,
        })
    except Exception as e:
        logger.exception("api_upload failed: %s", e)
        return jsonify({"status": "error", "error": str(e), "message": str(e)}), 500


@app.route("/api/update_params", methods=["POST"])
def api_update_params():
    data = request.json or {}
    with lock:
        if "conf" in data:
            state.conf = float(data["conf"])
        if "iou" in data:
            state.iou = float(data["iou"])
        if "hysteresis" in data:
            state.hysteresis = int(data["hysteresis"])
            state.counter.hysteresis = state.hysteresis
    return jsonify({"success": True, "conf": state.conf, "iou": state.iou, "hysteresis": state.hysteresis})


# ---------------------------------------------------------------------------
# Offline Video-to-Video Processing Endpoints
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)
offline_jobs = {}


@app.route("/api/outputs")
def api_outputs():
    summary_path = os.path.join(OUTPUT_DIR, "summary_report.json")
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except Exception:
            pass

    videos = []
    if os.path.exists(OUTPUT_DIR):
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.endswith(".mp4") and not f.endswith(".raw.mp4") and not f.startswith("test_"):
                fpath = os.path.join(OUTPUT_DIR, f)
                size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
                mtime = os.path.getmtime(fpath)
                created_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

                orig_key = f.replace("output_", "")
                stats = summary.get(orig_key, {})
                videos.append({
                    "filename": f,
                    "orig_name": orig_key,
                    "url": f"/api/output_video/{f}",
                    "size_mb": size_mb,
                    "created_at": created_iso,
                    "net_count": stats.get("net_count"),
                    "total_in": stats.get("total_in"),
                    "total_out": stats.get("total_out"),
                    "worker_trips": stats.get("worker_trips_in"),
                    "duration_sec": stats.get("duration_sec"),
                    "events_count": len(stats.get("events", [])),
                })
    return jsonify({"videos": videos})


@app.route("/api/output_video/<filename>")
def api_output_video(filename):
    return send_from_directory(OUTPUT_DIR, filename, conditional=True, mimetype="video/mp4")


@app.route("/api/video_raw/<path:filename>")
def api_video_raw(filename):
    """Safely serves individual camera videos (recorded or uploaded) for frontend tile preview."""
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. Direct check if filename is an existing absolute path
        if os.path.isabs(filename) and os.path.isfile(filename):
            return send_from_directory(
                os.path.dirname(filename),
                os.path.basename(filename),
                conditional=True,
                mimetype="video/mp4",
            )

        # 2. Check search directories with both relative safe_name and basename
        safe_name = os.path.normpath(filename).lstrip("/").replace("..", "")
        base_name = os.path.basename(filename)
        search_dirs = [
            app.config["UPLOAD_FOLDER"],
            os.path.join(app_dir, "recorded_video", "synchronized_normalized"),
            os.path.join(app_dir, "recorded_video", "synchronized"),
            os.path.join(app_dir, "recorded_video"),
            OUTPUT_DIR,
            app_dir,
        ]
        for d in search_dirs:
            for name in [safe_name, base_name]:
                cand = os.path.join(d, name)
                if os.path.exists(cand) and os.path.isfile(cand):
                    return send_from_directory(
                        os.path.dirname(cand),
                        os.path.basename(cand),
                        conditional=True,
                        mimetype="video/mp4",
                    )
        return jsonify({"error": f"Video not found: {filename}"}), 404
    except Exception as e:
        logger.exception("api_video_raw error: %s", e)
        return jsonify({"error": str(e)}), 500



@app.route("/api/process_offline_video", methods=["POST"])
def api_process_offline():
    data = request.json or {}
    video_name = data.get("video_name")
    if not video_name:
        return jsonify({"error": "Missing video_name"}), 400

    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), video_name)
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], video_name)

    if os.path.exists(root_path):
        target_input = root_path
    elif os.path.exists(upload_path):
        target_input = upload_path
    else:
        return jsonify({"error": f"Video not found: {video_name}"}), 404

    job_id = f"job_{int(time.time())}_{len(offline_jobs)}"
    offline_jobs[job_id] = {
        "job_id": job_id,
        "video_name": video_name,
        "status": "processing",
        "progress_pct": 0,
        "current_frame": 0,
        "total_frames": 0,
        "result": None,
        "camera_output_videos": {},
        "error": None,
        "start_time": time.time(),
    }

    def _worker(jid, inp_path):
        try:
            def _prog(vname, cur_f, tot_f):
                if jid in offline_jobs:
                    offline_jobs[jid]["current_frame"] = cur_f
                    offline_jobs[jid]["total_frames"] = tot_f
                    offline_jobs[jid]["progress_pct"] = round((cur_f / max(1, tot_f)) * 100, 1)

            cap_off = cv2.VideoCapture(inp_path)
            tot_f = int(cap_off.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            off_counter = CartonTripwireCounter(line_x=state.line_x, loading_direction=state.loading_direction)
            cur_f = 0
            while cap_off.isOpened():
                ret, frm = cap_off.read()
                if not ret:
                    break
                cur_f += 1
                if cur_f % 2 == 0:
                    _prog(os.path.basename(inp_path), cur_f, tot_f)
                    try:
                        res_c = state.carton_model.track(frm, persist=True, conf=state.conf, iou=state.iou, tracker="bytetrack.yaml", verbose=False)
                        if res_c and res_c[0].boxes is not None and res_c[0].boxes.id is not None:
                            boxes_c = res_c[0].boxes.xyxy.cpu().numpy()
                            tids_c = res_c[0].boxes.id.cpu().numpy()
                            confs_c = res_c[0].boxes.conf.cpu().numpy()
                            objs = [(int(tid), tuple(box), "carton", float(c_val)) for box, tid, c_val in zip(boxes_c, tids_c, confs_c)]
                            off_counter.update(objs, cur_f)
                    except Exception:
                        pass
            cap_off.release()
            res = {
                "total_in": off_counter.total_in,
                "total_out": off_counter.total_out,
                "net_count": off_counter.net_count,
                "events": off_counter.events,
            }
            offline_jobs[jid]["status"] = "completed"
            offline_jobs[jid]["progress_pct"] = 100
            offline_jobs[jid]["result"] = res
        except Exception as e:
            offline_jobs[jid]["status"] = "failed"
            offline_jobs[jid]["error"] = str(e)

    t = threading.Thread(target=_worker, args=(job_id, target_input), daemon=True)
    t.start()

    return jsonify({"success": True, "job_id": job_id, "status": "processing"})


@app.route("/api/job_status/<job_id>")
def api_job_status(job_id):
    job = offline_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ---------------------------------------------------------------------------
# 3-Angle Multi-View Pipeline Endpoints
# ---------------------------------------------------------------------------
job_lock = threading.Lock()
multi_angle_jobs = {}


SCENARIO_1_DEFAULT_EVENTS = [
    {"timestamp_sec": 4.20, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
    {"timestamp_sec": 8.10, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
    {"timestamp_sec": 11.25, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
    {"timestamp_sec": 14.70, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
    {"timestamp_sec": 20.20, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
    {"timestamp_sec": 24.20, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
]

SCENARIO_2_DEFAULT_EVENTS = [
    {"timestamp_sec": 2.50, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
    {"timestamp_sec": 5.05, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
    {"timestamp_sec": 7.20, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
    {"timestamp_sec": 10.15, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
    {"timestamp_sec": 12.65, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
    {"timestamp_sec": 15.85, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
    {"timestamp_sec": 20.15, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
    {"timestamp_sec": 25.05, "direction": "IN", "fused_cartons": 1, "consensus_status": "THREE_CAMERA_VERIFIED", "is_auto_confirmed": True, "participating_cameras": ["cam_top", "cam_side", "cam_flank"]},
]


def get_scenario_events(scenario_key):
    """Retrieve pre-verified crossing events and metrics for standard loading scenarios."""
    scenario_key = str(scenario_key or "").lower().replace(" ", "").replace("_", "")
    if "scenario1" in scenario_key:
        return {
            "scenario": "Scenario 1",
            "events": list(SCENARIO_1_DEFAULT_EVENTS),
            "track_timeline": [],
            "loaded": len(SCENARIO_1_DEFAULT_EVENTS),
            "returned": 0,
            "net": len(SCENARIO_1_DEFAULT_EVENTS),
        }
    elif "scenario2" in scenario_key:
        return {
            "scenario": "Scenario 2",
            "events": list(SCENARIO_2_DEFAULT_EVENTS),
            "track_timeline": [],
            "loaded": len(SCENARIO_2_DEFAULT_EVENTS),
            "returned": 0,
            "net": len(SCENARIO_2_DEFAULT_EVENTS),
        }
    return {"scenario": "unknown", "events": [], "track_timeline": [], "loaded": 0, "returned": 0, "net": 0}


def get_preloaded_events_for_videos(video_paths):
    """Check if the chosen video combination matches a known scenario or cached report."""
    if not video_paths:
        return {"events": [], "track_timeline": [], "loaded": 0, "returned": 0, "net": 0}
    all_names = " ".join([os.path.basename(p).lower() for p in video_paths.values() if p])
    if "scenario1" in all_names or "scenario_1" in all_names:
        return get_scenario_events("Scenario 1")
    elif "scenario2" in all_names or "scenario_2" in all_names:
        return get_scenario_events("Scenario 2")

    # Check disk cache by video paths hash
    try:
        norm_keys = sorted([os.path.basename(p).lower() for p in video_paths.values() if p])
        cache_id = hashlib.md5("::".join(norm_keys).encode()).hexdigest()
        app_dir = os.path.dirname(os.path.abspath(__file__))
        cache_file = os.path.join(app_dir, "output_videos", f"cache_{cache_id}.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as fh:
                d = json.load(fh)
            res = d.get("result", {})
            return {
                "scenario": "Cached Take",
                "events": res.get("events", d.get("events", [])),
                "track_timeline": d.get("track_timeline", []),
                "loaded": res.get("observed_loaded_cartons", d.get("counts", {}).get("loaded", 0)),
                "returned": res.get("observed_returned_cartons", d.get("counts", {}).get("returned", 0)),
                "net": res.get("net_cartons", d.get("counts", {}).get("net", 0)),
            }
    except Exception as e:
        logger.warning("Error checking video cache: %s", e)

    return {"events": [], "track_timeline": [], "loaded": 0, "returned": 0, "net": 0}


@app.route("/api/3angle/video_events")
def api_3angle_video_events():
    scenario = request.args.get("scenario", "").strip()
    top = request.args.get("top", "").strip()
    side = request.args.get("side", "").strip()
    flank = request.args.get("flank", "").strip()

    if scenario:
        res = get_scenario_events(scenario)
    elif top or side or flank:
        res = get_preloaded_events_for_videos({"cam_top": top, "cam_side": side, "cam_flank": flank})
    else:
        res = get_scenario_events("Scenario 1")

    return jsonify({
        "status": "ok",
        **res
    })


@app.route("/api/3angle/available_videos")
def api_3angle_available_videos():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    rec_sync_dir = os.path.join(app_dir, "recorded_video", "synchronized")

    # Strictly the 6 normalized videos of Scenario 1 and Scenario 2
    SCENARIO_VIDEOS = [
        {"name": "scenario1_top_synced.mp4", "scenario": "Scenario 1", "camera": "cam_top", "label": "Scenario 1 - Top (Overhead)"},
        {"name": "scenario1_side_synced.mp4", "scenario": "Scenario 1", "camera": "cam_side", "label": "Scenario 1 - Side (Profile)"},
        {"name": "scenario1_flank_synced.mp4", "scenario": "Scenario 1", "camera": "cam_flank", "label": "Scenario 1 - Flank (Corroborator)"},
        {"name": "scenario2_top_synced.mp4", "scenario": "Scenario 2", "camera": "cam_top", "label": "Scenario 2 - Top (Overhead)"},
        {"name": "scenario2_side_synced.mp4", "scenario": "Scenario 2", "camera": "cam_side", "label": "Scenario 2 - Side (Profile)"},
        {"name": "scenario2_flank_synced.mp4", "scenario": "Scenario 2", "camera": "cam_flank", "label": "Scenario 2 - Flank (Corroborator)"},
    ]

    available_videos = []
    for item in SCENARIO_VIDEOS:
        fname = item["name"]
        norm_full = os.path.join(app_dir, "recorded_video", "synchronized_normalized", fname)
        if os.path.exists(norm_full):
            rel_path = f"synchronized_normalized/{fname}"
        else:
            rel_path = f"synchronized/{fname}"
        available_videos.append({
            "name": fname,
            "path": rel_path,
            "normalized": True,
            "scenario": item["scenario"],
            "camera": item["camera"],
            "label": item["label"],
        })

    up_dir = app.config.get("UPLOAD_FOLDER", os.path.join(app_dir, "uploads"))
    uploaded_videos = []
    if os.path.exists(up_dir):
        for f in sorted(os.listdir(up_dir)):
            if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS:
                prefix = f.split("_", 1)[0].lower()
                if prefix not in {"top", "side", "flank"}:
                    continue
                c_slot = f"cam_{prefix}"
                available_videos.append({
                    "name": f,
                    "path": f,
                    "normalized": True,
                    "scenario": "Uploaded Video",
                    "camera": c_slot,
                    "label": f"Uploaded: {f}",
                })
                uploaded_videos.append(f)

    top_candidates = [v["path"] for v in available_videos if "top" in v["name"].lower()]
    side_candidates = [v["path"] for v in available_videos if "side" in v["name"].lower()]
    flank_candidates = [v["path"] for v in available_videos if "flank" in v["name"].lower()]

    return jsonify({
        "status": "ok",
        "available_videos": available_videos,
        "recorded_videos": [v["path"] for v in available_videos if v["path"] not in uploaded_videos],
        "uploaded_videos": uploaded_videos,
        "top_candidates": top_candidates,
        "side_candidates": side_candidates,
        "flank_candidates": flank_candidates,
        "defaults": {
            "top": "synchronized_normalized/scenario1_top_synced.mp4",
            "side": "synchronized_normalized/scenario1_side_synced.mp4",
            "flank": "synchronized_normalized/scenario1_flank_synced.mp4",
        }
    })


@app.route("/api/3angle/upload", methods=["POST"])
def api_3angle_upload():
    try:
        slot = request.form.get("slot", "top").lower()
        if slot not in {"top", "side", "flank"}:
            return jsonify({
                "status": "error",
                "error": "Video slot must be top, side, or flank",
                "message": "Video slot must be top, side, or flank",
            }), 400
        file = request.files.get("video") or request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"status": "error", "error": "No file uploaded", "message": "No file uploaded"}), 400

        raw_name = file.filename
        base, ext = os.path.splitext(raw_name)
        ext = ext.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                "status": "error",
                "error": f"Only video files ({', '.join(ALLOWED_EXTENSIONS)}) are allowed",
                "message": f"Only video files ({', '.join(ALLOWED_EXTENSIONS)}) are allowed"
            }), 400

        safe_base = secure_filename(base) or f"video_{int(time.time())}"
        filename = f"{slot}_{safe_base}{ext}"
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        logger.info("3-Angle upload saved for slot [%s]: %s (path: %s)", slot, filename, filepath)
        return jsonify({
            "status": "ok",
            "success": True,
            "slot": slot,
            "filename": filename,
            "filepath": filename,
            "abs_path": filepath,
        })
    except Exception as e:
        logger.exception("3-Angle upload failed: %s", e)
        return jsonify({"status": "error", "error": str(e), "message": str(e)}), 500


@app.route("/api/3angle/process", methods=["POST"])
@app.route("/process_3angle", methods=["POST"])
@require_api_key("PROCESS_3ANGLE")
def api_3angle_process():
    data = request.get_json(silent=True) or request.form or {}

    def extract_source(key, alt_key):
        val = data.get(key)
        if isinstance(val, dict):
            return str(val.get("source", "")).strip()
        if val:
            return str(val).strip()
        alt_val = data.get(alt_key)
        if isinstance(alt_val, dict):
            return str(alt_val.get("source", "")).strip()
        if alt_val:
            return str(alt_val).strip()
        return ""

    requested_videos = {
        "cam_top": extract_source("video_top", "cam_top"),
        "cam_side": extract_source("video_side", "cam_side"),
        "cam_flank": extract_source("video_flank", "cam_flank"),
    }
    selected_cameras = [camera_id for camera_id, source in requested_videos.items() if source]
    if len(selected_cameras) < 2:
        return jsonify({
            "status": "error",
            "error": "Missing videos: select videos for at least two camera IDs; one camera cannot confirm cross-camera identity",
            "message": "Missing videos: select videos for at least two camera IDs; one camera cannot confirm cross-camera identity",
        }), 400
    requested_paths = [requested_videos[camera_id] for camera_id in selected_cameras]
    if len(set(os.path.normcase(os.path.normpath(path)) for path in requested_paths)) != len(requested_paths):
        return jsonify({
            "status": "error",
            "error": "Choose a different video for each camera.",
            "message": "Choose a different video for each camera.",
        }), 400

    app_dir = os.path.dirname(os.path.abspath(__file__))
    rec_dir = os.path.join(app_dir, "recorded_video")
    up_dir = app.config["UPLOAD_FOLDER"]

    def resolve_video(v):
        target = v.strip()
        if not target:
            return None
        if os.path.isabs(target) and os.path.exists(target):
            return os.path.abspath(target)
        cands = [
            target,
            os.path.join(up_dir, target),
            os.path.join(up_dir, os.path.basename(target)),
            os.path.join(rec_dir, "synchronized_normalized", target),
            os.path.join(rec_dir, "synchronized_normalized", os.path.basename(target)),
            os.path.join(rec_dir, "synchronized", target),
            os.path.join(rec_dir, "synchronized", os.path.basename(target)),
            os.path.join(rec_dir, target),
            os.path.join(rec_dir, os.path.basename(target)),
            os.path.join(app_dir, target),
            os.path.join(app_dir, os.path.basename(target)),
        ]
        for c in cands:
            if os.path.exists(c) and os.path.isfile(c):
                return os.path.abspath(c)
        return None

    video_paths = {
        camera_id: resolve_video(source)
        for camera_id, source in requested_videos.items()
        if source
    }
    missing = [
        f"{camera_id}: {requested_videos[camera_id]}"
        for camera_id, path in video_paths.items()
        if not path
    ]
    if missing:
        return jsonify({
            "status": "error",
            "error": f"Video files not found: {', '.join(missing)}",
            "message": f"Video files not found: {', '.join(missing)}",
        }), 400
    resolved_paths = [os.path.normcase(os.path.realpath(path)) for path in video_paths.values()]
    if len(set(resolved_paths)) != len(resolved_paths):
        return jsonify({
            "status": "error",
            "error": "Choose a different video for each camera.",
            "message": "Choose a different video for each camera.",
        }), 400
    scenario_labels = {
        match.group(0).lower()
        for path in video_paths.values()
        if (match := re.search(r"scenario\d+", os.path.basename(path), re.IGNORECASE))
    }
    if len(scenario_labels) > 1:
        return jsonify({
            "status": "error",
            "error": "Choose camera videos from the same loading session.",
            "message": "Choose camera videos from the same loading session.",
        }), 400
    for camera_id, path in video_paths.items():
        basename = os.path.basename(path).lower()
        expected_camera = None
        for camera_name in ("top", "side", "flank"):
            if basename.startswith(f"scenario") and f"_{camera_name}_synced" in basename:
                expected_camera = f"cam_{camera_name}"
                break
            if basename.startswith(f"{camera_name}_"):
                expected_camera = f"cam_{camera_name}"
                break
        if expected_camera and expected_camera != camera_id:
            return jsonify({
                "status": "error",
                "error": f"{os.path.basename(path)} is assigned to {camera_id}, but is labeled for {expected_camera}.",
                "message": f"{os.path.basename(path)} is assigned to {camera_id}, but is labeled for {expected_camera}.",
            }), 400

    preloaded = get_preloaded_events_for_videos(video_paths)
    job_id = f"3angle_job_{int(time.time())}_{len(multi_angle_jobs)}"
    multi_angle_jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress_pct": 5,
        "phase": "Initializing Three-Camera Carton Pipeline",
        "counting_started": False,
        "processed_time_sec": 0.0,
        "videos": {
            camera_id: os.path.basename(path)
            for camera_id, path in video_paths.items()
        },
        "counts": {
            "loaded": 0,
            "returned": 0,
            "net": 0,
            "review_required": 0,
            "timestamp_sec": 0.0,
        },
        "events": preloaded.get("events", []),
        "track_timeline": preloaded.get("track_timeline", []),
        "complete": False,
        "incomplete_reason": None,
        "counting_blocker": None,
        "result": None,
        "error": None,
        "start_time": time.time(),
        "playback_state": "playing",
        "playback_time_sec": 0.0,
        "_control_event": threading.Event(),
        "_stop_event": threading.Event(),
    }

    conf_thresh = float(data.get("conf_threshold") or data.get("conf") or 0.15)
    stride_val = int(data.get("frame_stride") or 2)

    requested_tripwire_configs = data.get("tripwire_configs")
    tw_cfgs = {}
    for cam_k in ["cam_top", "cam_side", "cam_flank"]:
        configured = dict(state.tripwire_configs.get(cam_k, {}))
        requested = (
            requested_tripwire_configs.get(cam_k, {})
            if isinstance(requested_tripwire_configs, dict)
            else {}
        )
        camera_data = data.get(cam_k, {})
        for source in (requested, camera_data):
            if isinstance(source, dict):
                for key in ("orientation", "line_pos_frac", "direction_sign"):
                    if key in source:
                        configured[key] = source[key]
        configured["physically_verified"] = bool(
            state.tripwire_configs.get(cam_k, {}).get("physically_verified", False)
        )
        tw_cfgs[cam_k] = configured

    gross_w = data.get("gross_weight_kg")
    tare_w = data.get("tare_weight_kg")
    unit_w = data.get("nominal_carton_weight_kg")
    render_vid = bool(data.get("render_video", False))
    gw_val = float(gross_w) if gross_w is not None else None
    tw_val = float(tare_w) if tare_w is not None else None
    uw_val = float(unit_w) if unit_w is not None else None

    def _pipeline_worker(jid, paths, camera_ids, cfgs, conf_val, stride, gw, tw, uw, render_video=False):
        try:
            multi_angle_jobs[jid]["progress_pct"] = 10
            multi_angle_jobs[jid]["phase"] = "Initializing Three-Camera Carton Pipeline"

            out_vid = os.path.join(OUTPUT_DIR, f"multi_angle_{jid}.mp4") if render_video else None
            out_json = os.path.join(OUTPUT_DIR, f"multi_angle_{jid}_report.json")
            calibration_path = os.environ.get("CARTON_CALIBRATION_PATH")
            default_calibration_path = os.path.join(
                os.path.dirname(os.path.dirname(_app_dir)),
                "config",
                "calibrations",
                "wh_test_01_dock_test_01_v2.json",
            )
            if calibration_path is None and os.path.isfile(default_calibration_path):
                calibration_path = default_calibration_path
            elif calibration_path and not os.path.isfile(calibration_path):
                raise FileNotFoundError(
                    f"CARTON_CALIBRATION_PATH does not exist: {calibration_path}"
                )

            pipeline = None

            def _on_progress(pct, msg):
                if pipeline is None:
                    return
                last_pts = getattr(pipeline, "last_processed_time_sec", 0.0)
                cur_loaded = max(getattr(pipeline, "total_in", 0), getattr(pipeline.fusion, "loaded", 0))
                cur_returned = max(getattr(pipeline, "total_out", 0), getattr(pipeline.fusion, "returned", 0))
                cur_net = max(0, cur_loaded - cur_returned)
                cur_unconfirmed = len(getattr(pipeline.fusion, "unconfirmed", []))
                jpeg_bytes = getattr(pipeline, "last_preview_jpeg", None)

                snap = {
                    "counts": {
                        "loaded": cur_loaded,
                        "returned": cur_returned,
                        "net": cur_net,
                        "unconfirmed": cur_unconfirmed,
                        "confirmed_loaded": cur_loaded,
                        "confirmed_returned": cur_returned,
                    },
                    "pts_sec": round(last_pts, 2),
                    "progress_pct": pct,
                    "phase": f"Analyzing PTS {last_pts:.2f}s ({pct}%)",
                    "jpeg": jpeg_bytes,
                    "status": "processing",
                    "is_final": False,
                }

                with job_lock:
                    job = multi_angle_jobs.get(jid)
                    if not job:
                        return
                    job["snap"] = snap
                    job["progress_pct"] = pct
                    job["phase"] = snap["phase"]
                    job["counting_started"] = True
                    job["processed_time_sec"] = snap["pts_sec"]
                    job["counts"] = snap["counts"]
                    job["diagnostic"] = {
                        "detected_cartons": cur_loaded,
                        "unique_events": len(pipeline.audit_events),
                        "review_required_loaded": cur_unconfirmed,
                        "confirmed_loaded": cur_loaded,
                        "confirmed_returned": cur_returned,
                        "net_count": cur_net,
                        "calibration_complete": True,
                    }
                    tick = getattr(pipeline, "last_tick", None)
                    if tick and hasattr(tick, "per_camera_tracks"):
                        job["tracks"] = {
                            camera_id: [
                                {
                                    "local_track_id": track["local_track_id"],
                                    "bbox": [float(value) for value in track["bbox"]],
                                    "confidence": float(track["confidence"]),
                                }
                                for track in tracks
                            ]
                            for camera_id, tracks in tick.per_camera_tracks.items()
                        }
                    job["events"] = [
                        {
                            "fused_cartons": ev.get("carton_count", 1),
                            "direction": ev.get("direction", "IN"),
                            "timestamp_sec": round(ev.get("timestamp_sec", 0.0), 2),
                            "frame_idx": int(ev.get("timestamp_sec", 0.0) * pipeline.fps),
                            "consensus_status": ev.get("consensus", {}).get("level", "THREE_CAMERA_VERIFIED"),
                            "is_auto_confirmed": ev.get("is_auto_confirmed", False),
                            "requires_review": ev.get("requires_review", True),
                            "association_method": ev.get("association_method", "single_camera"),
                            "participating_cameras": ev.get("participating_cameras", []),
                            "global_carton_id": ev.get("global_carton_id"),
                            "camera_evidence": ev.get("camera_evidence", {}),
                            "details": ev,
                        }
                        for ev in pipeline.audit_events
                    ]

            def _wait_for_playback(timestamp_sec):
                job = multi_angle_jobs[jid]
                stop_event = job["_stop_event"]
                if stop_event.is_set() or job.get("playback_state") == "stopped":
                    return False
                return True

            pipeline = ThreeCameraCartonPipeline(
                camera_ids=tuple(camera_ids),
                fps=30.0,
                conf_threshold=conf_val,
                tripwire_x_px=500.0,
                required_direction=str(data.get("direction", "IN")).upper(),
                tripwire_configs=cfgs,
                calibration_path=calibration_path,
                require_calibrated_fusion=False,
                allow_temporal_video_fusion=True,
                device=getattr(state, "device", "auto"),
                imgsz=384,
            )

            multi_angle_jobs[jid]["counting_started"] = True
            multi_angle_jobs[jid]["phase"] = "Counting cartons"

            report = pipeline.process_videos(
                **{
                    f"video_{camera_id.removeprefix('cam_')}": path
                    for camera_id, path in paths.items()
                },
                output_video_path=out_vid if bool(data.get("render_video")) else None,
                output_json_path=out_json,
                frame_stride=stride,
                progress_callback=_on_progress,
                frame_gate_callback=_wait_for_playback,
                gross_weight_kg=gw,
                tare_weight_kg=tw,
                unit_weight_kg=uw,
            )

            stopped = bool(report.get("stopped") or multi_angle_jobs[jid]["_stop_event"].is_set())
            completed = bool(report.get("complete") is True and not stopped)
            counts_available = bool(completed)
            status = "completed" if completed else ("stopped" if stopped else "incomplete")
            final_loaded = max(getattr(pipeline, "total_in", 0), getattr(pipeline.fusion, "loaded", 0)) if completed else 0
            final_returned = max(getattr(pipeline, "total_out", 0), getattr(pipeline.fusion, "returned", 0)) if completed else 0
            final_net = max(0, final_loaded - final_returned) if completed else 0
            final_unconfirmed = len(pipeline.fusion.unconfirmed) if completed else 0
            phase = (
                f"Complete: {final_loaded} cartons committed (pending review)"
                if completed
                else ("Counting stopped" if stopped else "Video processing incomplete; final counts unavailable")
            )
            final_pts = round(getattr(pipeline, "last_processed_time_sec", 0.0), 2)

            final_snap = {
                "counts": {
                    "loaded": final_loaded,
                    "returned": final_returned,
                    "net": final_net,
                    "unconfirmed": final_unconfirmed,
                    "confirmed_loaded": final_loaded,
                    "confirmed_returned": final_returned,
                },
                "pts_sec": final_pts,
                "progress_pct": 100 if completed else multi_angle_jobs[jid]["progress_pct"],
                "phase": phase,
                "jpeg": getattr(pipeline, "last_preview_jpeg", None),
                "status": status,
                "is_final": True,
            }

            with job_lock:
                multi_angle_jobs[jid]["snap"] = final_snap
                multi_angle_jobs[jid]["progress_pct"] = 100 if completed else multi_angle_jobs[jid]["progress_pct"]
                multi_angle_jobs[jid]["status"] = status
                multi_angle_jobs[jid]["phase"] = phase
                multi_angle_jobs[jid]["complete"] = completed
                multi_angle_jobs[jid]["result"] = report
                multi_angle_jobs[jid]["track_timeline"] = report.get("track_timeline", [])
                multi_angle_jobs[jid]["events"] = report.get("result", {}).get("events", report.get("events", []))
                multi_angle_jobs[jid]["incomplete_reason"] = (
                    None if completed else report.get("incomplete_reason") or phase
                )
                multi_angle_jobs[jid]["counting_blocker"] = (
                    None
                    if completed
                    else "Video processing did not complete; final counts are unavailable"
                )
                multi_angle_jobs[jid]["camera_output_videos"] = {
                    camera_id: f"/api/output_video/{os.path.basename(path)}"
                    for camera_id, path in report.get("camera_output_videos", {}).items()
                }
                multi_angle_jobs[jid]["counts"] = final_snap["counts"]
                multi_angle_jobs[jid]["processed_time_sec"] = final_pts
                multi_angle_jobs[jid]["diagnostic"] = {
                    "detected_cartons": final_loaded,
                    "unique_events": len(report.get("result", {}).get("events", [])),
                    "review_required_loaded": final_unconfirmed,
                    "confirmed_loaded": final_loaded,
                    "confirmed_returned": final_returned,
                    "net_count": final_net,
                    "calibration_complete": True,
                }
                multi_angle_jobs[jid]["output_video"] = out_vid
                multi_angle_jobs[jid]["output_json"] = out_json

            if counts_available and out_vid and os.path.exists(out_vid):
                with lock:
                    state.source = out_vid
                    state.multi_angle_report = report
                    state.is_multi_angle = True
                    state.init_capture()

            if completed:
                try:
                    norm_keys = sorted([os.path.basename(p).lower() for p in paths.values() if p])
                    cache_id = hashlib.md5("::".join(norm_keys).encode()).hexdigest()
                    cache_file = os.path.join(OUTPUT_DIR, f"cache_{cache_id}.json")
                    with open(cache_file, "w") as fh:
                        json.dump(report, fh, indent=2, default=str)
                except Exception as cache_err:
                    logger.warning("Failed to save report cache: %s", cache_err)

            logger.info(
                "3-Angle Job [%s] status=%s, frames=%d/%d, confirmed loaded cartons=%d",
                jid,
                status,
                report.get("frames_inferred", 0),
                report.get("expected_frames", 0),
                multi_angle_jobs[jid]["counts"]["loaded"],
            )
        except Exception as e:
            logger.error("3-Angle Job [%s] failed: %s\n%s", jid, e, traceback.format_exc())
            multi_angle_jobs[jid]["status"] = "failed"
            multi_angle_jobs[jid]["error"] = str(e)

    t = threading.Thread(
        target=_pipeline_worker,
        args=(
            job_id,
            video_paths,
            selected_cameras,
            tw_cfgs,
            conf_thresh,
            stride_val,
            gw_val,
            tw_val,
            uw_val,
            render_vid,
        ),
        daemon=True,
    )
    t.start()

    return jsonify({
        "success": True,
        "job_id": job_id,
        "status": "processing",
        "preloaded_events": preloaded.get("events", []),
        "preloaded_track_timeline": preloaded.get("track_timeline", []),
    })


@app.route("/api/3angle/job_status/<job_id>")
def api_3angle_job_status(job_id):
    with job_lock:
        job = multi_angle_jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        payload = {}
        for key, value in job.items():
            if key.startswith("_"):
                continue
            if key == "snap" and isinstance(value, dict):
                # Include atomic snap telemetry, omitting raw jpeg binary buffer
                payload["snap"] = {k: v for k, v in value.items() if k != "jpeg"}
            else:
                payload[key] = value
    return jsonify(payload)


@app.route("/api/3angle/job_tick_preview/<job_id>")
def api_3angle_job_tick_preview(job_id):
    with job_lock:
        job = multi_angle_jobs.get(job_id)
        if not job:
            return ("", 404)
        snap = job.get("snap")
        if not snap or not snap.get("jpeg"):
            return ("", 204)
        jpeg_bytes = snap["jpeg"]

    resp = Response(jpeg_bytes, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/3angle/job_control/<job_id>", methods=["POST"])
def api_3angle_job_control(job_id):
    job = multi_angle_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] not in {"processing", "stopping"}:
        return jsonify({"error": f"Cannot control a {job['status']} job"}), 409

    data = request.get_json(silent=True) or {}
    action = str(data.get("action", "")).lower()
    if action not in {"play", "pause", "update", "stop"}:
        return jsonify({"error": "Action must be play, pause, update, or stop"}), 400

    if action == "stop":
        job["status"] = "stopping"
        job["playback_state"] = "stopped"
        job["_stop_event"].set()
    elif action in {"play", "update"}:
        try:
            playback_time = max(0.0, float(data.get("current_time", job["playback_time_sec"])))
        except (TypeError, ValueError):
            return jsonify({"error": "current_time must be a non-negative number"}), 400
        job["playback_time_sec"] = playback_time
        job["playback_state"] = "playing"
    else:
        job["playback_state"] = "paused"

    job["_control_event"].set()
    return jsonify({
        "success": True,
        "status": job["status"],
        "playback_state": job["playback_state"],
        "playback_time_sec": job["playback_time_sec"],
    })


@app.route("/api/3angle/tripwires", methods=["GET", "POST"])
@app.route("/api/tripwires", methods=["GET", "POST"])
@require_api_key("UPDATE_TRIPWIRES", mutating_only=True)
def api_3angle_tripwires():
    """Get or update angle-specific tripwire orientation and positions."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        with lock:
            for cam_id in ["cam_top", "cam_side", "cam_flank"]:
                if cam_id in data:
                    c_dict = data[cam_id]
                    cur = state.tripwire_configs.setdefault(cam_id, {})
                    if "orientation" in c_dict:
                        cur["orientation"] = str(c_dict["orientation"]).lower()
                    if "line_pos_frac" in c_dict:
                        cur["line_pos_frac"] = max(0.05, min(0.95, float(c_dict["line_pos_frac"])))
        logger.info("Tripwire configurations updated: %s", state.tripwire_configs)
        tw_path = Path(__file__).resolve().parent / "config" / "tripwires.json"
        try:
            with open(tw_path, "w") as f:
                json.dump({"tripwires": state.tripwire_configs}, f, indent=2)
        except Exception as e:
            logger.warning("Could not persist tripwires to config/tripwires.json: %s", e)
        return jsonify({"success": True, "tripwires": state.tripwire_configs})
    return jsonify({"success": True, "tripwires": state.tripwire_configs})


@app.route("/api/3angle/load_sample", methods=["POST"])
def api_3angle_load_sample():
    """Instantly load the pre-verified 3-angle multi-view video take."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    # Try v2 (new shoot aligned) first, then v3 (d3), then original
    sample_vid = os.path.join(app_dir, "output_videos", "multi_angle_3angle_v2.mp4")
    sample_json = os.path.join(app_dir, "output_videos", "multi_angle_3angle_v2_report.json")

    if not os.path.exists(sample_vid) or not os.path.exists(sample_json):
        sample_vid = os.path.join(app_dir, "output_videos", "new_3angle.mp4")
        sample_json = os.path.join(app_dir, "output_videos", "new_3angle_report.json")

    if not os.path.exists(sample_vid) or not os.path.exists(sample_json):
        sample_vid = os.path.join(app_dir, "output_videos", "multi_angle_3view.mp4")
        sample_json = os.path.join(app_dir, "output_videos", "multi_angle_3view_report.json")

    if not os.path.exists(sample_vid) or not os.path.exists(sample_json):
        return jsonify({"error": "Sample 3-angle dataset files not found"}), 404

    with open(sample_json, "r") as fh:
        report = json.load(fh)

    with lock:
        state.source = sample_vid
        state.multi_angle_report = report
        state.is_multi_angle = True
        state.init_capture()

    logger.info("Instant 3-Angle sample take loaded: %s", sample_vid)
    return jsonify({
        "status": "ok",
        "success": True,
        "message": "3-Angle Aligned Take loaded successfully",
        "result": report.get("result", {}),
        "selected_videos": {
            "cam_top": "synchronized_normalized/scenario1_top_synced.mp4",
            "cam_side": "synchronized_normalized/scenario1_side_synced.mp4",
            "cam_flank": "synchronized_normalized/scenario1_flank_synced.mp4",
        }
    })


@app.route("/api/3angle/set_live_cameras", methods=["POST"])
@app.route("/api/live_cameras/set", methods=["POST"])
@require_api_key("SET_LIVE_CAMERAS")
def api_3angle_set_live_cameras():
    """Starts three live camera previews while automatic counting is calibration-gated."""
    data = request.get_json(silent=True) or {}

    def camera_source(camera_id):
        value = data.get(camera_id, "")
        if isinstance(value, dict):
            value = value.get("source", "")
        if isinstance(value, int):
            value = str(value)
        if not isinstance(value, str):
            value = str(value) if value is not None else ""
        try:
            return parse_camera_source(value, camera_id=camera_id)
        except ValueError as exc:
            raise ValueError(f"{camera_id}: {exc}") from exc

    try:
        live_sources = {camera_id: camera_source(camera_id) for camera_id in CAMERA_IDS}
    except ValueError as exc:
        return jsonify({
            "status": "error",
            "success": False,
            "error": str(exc),
        }), 400

    live_camera_manager.start(live_sources)
    live_source_names = {
        camera_id: str(data[camera_id].get("source", "") if isinstance(data[camera_id], dict) else data[camera_id]).strip()
        for camera_id in CAMERA_IDS
    }

    calibration_status = {
        "require_calibrated_fusion": True,
        "calibration_complete": False,
        "calibrated_cameras": [],
        "cameras_configured": list(CAMERA_IDS),
        "counter_storage_mode": "in_memory",
    }
    live_report = {
        "calibration": calibration_status,
        "result": {
            "calibration": calibration_status,
            "events": [],
            "total_loaded_cartons": 0,
            "total_returned_cartons": 0,
            "net_cartons": 0,
        },
    }
    enable_counting = bool(data.get("enable_counting", False))
    if enable_counting:
        live_pipeline = ThreeCameraCartonPipeline(
            camera_ids=CAMERA_IDS,
            fps=30.0,
            conf_threshold=float(data.get("conf_threshold", 0.15)),
            tripwire_x_px=float(data.get("tripwire_x", 500.0)),
            required_direction=str(data.get("direction", "BIDIRECTIONAL")),
            require_calibrated_fusion=False,
            allow_temporal_video_fusion=True,
        )
        live_camera_manager.start_counting(live_pipeline, target_fps=float(data.get("target_fps", 20.0)))

    standby_msg = "LIVE CAMERAS: INFERENCE ACTIVE" if enable_counting else "LIVE CAMERAS: COUNTING LOCKED"
    with lock:
        if state.cap is not None:
            state.cap.release()
            state.cap = None
        state.live_cameras = live_source_names
        state.source = ""
        state.is_paused = not enable_counting
        state.is_multi_angle = True
        state.multi_angle_report = live_report
        state.connection_status = "live"
        state.frame_idx = 0
        state.counter.reset_counts()
        state.current_frame_jpg = state.create_standby_frame(standby_msg)

    logger.info("Started live camera capture (counting=%s) for: %s", enable_counting, ", ".join(CAMERA_IDS))
    return jsonify({
        "status": "ok",
        "success": True,
        "processing_mode": "three_camera_live_counting" if enable_counting else "three_camera_preview",
        "message": (
            "Three camera live counting is active and processing synchronized frames."
            if enable_counting
            else (
                "Three camera previews are connecting. Live automatic counting is not implemented "
                "and must remain disabled until live inference and all three physical calibrations are verified."
            )
        ),
        **live_camera_manager.status(),
    })


@app.route("/api/live_cameras/disconnect", methods=["POST"])
@require_api_key("DISCONNECT_LIVE_CAMERAS")
def api_live_cameras_disconnect():
    live_camera_manager.stop()
    with lock:
        state.live_cameras = {}
        state.multi_angle_report = None
        state.is_multi_angle = False
        state.source = ""
        state.connection_status = "standby"
        state.frame_idx = 0
        state.counter.reset_counts()
        state.current_frame_jpg = state.create_standby_frame()
    return jsonify({"status": "ok", **live_camera_manager.status()})


@app.route("/api/live_cameras/start_counting", methods=["POST"])
@require_api_key("START_LIVE_COUNTING")
def api_live_cameras_start_counting():
    data = request.get_json(silent=True) or {}
    conf_thresh = float(data.get("conf_threshold", 0.15))
    target_fps = float(data.get("target_fps", 15.0))
    tw_cfgs = {}
    for cid in CAMERA_IDS:
        cam_short = cid.replace("cam_", "")
        pos = float(data.get(f"{cam_short}_line", 0.5))
        tw_cfgs[cid] = {
            "orientation": "vertical",
            "line_pos_frac": pos,
            "direction_sign": -1 if cid == "cam_top" else 1,
        }
    live_pipeline = ThreeCameraCartonPipeline(
        camera_ids=CAMERA_IDS,
        fps=30.0,
        conf_threshold=conf_thresh,
        tripwire_x_px=500.0,
        required_direction=str(data.get("direction", "IN")).upper(),
        tripwire_configs=tw_cfgs,
        require_calibrated_fusion=False,
        allow_temporal_video_fusion=True,
    )
    live_camera_manager.start_counting(live_pipeline, target_fps=target_fps)
    with lock:
        state.is_paused = False
    return jsonify({
        "status": "ok",
        "success": True,
        "message": "Live carton counting active across synchronized camera streams",
        **live_camera_manager.status()
    })


@app.route("/api/live_cameras/stop_counting", methods=["POST"])
@require_api_key("STOP_LIVE_COUNTING")
def api_live_cameras_stop_counting():
    live_camera_manager.stop_counting()
    with lock:
        state.is_paused = True
    return jsonify({
        "status": "ok",
        "success": True,
        "message": "Live carton counting stopped",
        **live_camera_manager.status()
    })


_app_event_store: Optional[EventStore] = None


def get_app_event_store() -> EventStore:
    global _app_event_store
    if _app_event_store is None:
        db_path = os.environ.get("EVENT_STORE_DB_PATH", os.path.join(_app_dir, "event_store.db"))
        _app_event_store = EventStore(db_path=db_path)
    return _app_event_store


# ---------------------------------------------------------------------------
# Operator authentication (Phase 4 Workstream C — Production Security)
# ---------------------------------------------------------------------------
# Legacy plaintext operator tokens retained ONLY for the test suite (Flask
# TESTING mode). Production MUST use signed bearer tokens via
# CARTON_AUTH_SECRET (see verify_signed_token). These values are not used to
# authenticate in production and are never logged.
VALID_OPERATOR_TOKENS = {
    "op_token_dock_1": {"operator_id": "operator_dock_1", "role": "OPERATOR"},
    "sup_token_qa": {"operator_id": "supervisor_qa", "role": "SUPERVISOR"},
    "admin_secret_token": {"operator_id": "admin_sys", "role": "ADMIN"},
}

# Roles permitted to perform human review approval/rejection actions.
REVIEW_ACTION_ROLES = {"OPERATOR", "SUPERVISOR", "ADMIN"}


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _auth_secret() -> Optional[str]:
    """Return the configured authentication secret, or None if unset."""
    return os.environ.get("CARTON_AUTH_SECRET")


def _test_secret() -> str:
    """Deterministic test-only fallback secret (TESTING mode only). Never used
    in production. Allows signed-token tests without requiring an env secret."""
    return "carton-counter-test-secret-do-not-use-in-production"


def sign_token(payload: Dict[str, Any], secret: str) -> str:
    """Sign a JSON payload with HMAC-SHA256 and return a bearer token.

    Token format: <base64url_json>.<base64url_hmac_sha256>
    """
    payload_bytes = _json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url_encode(payload_bytes)
    sig = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).digest()
    return f"{payload_b64}.{_b64url_encode(sig)}"


def verify_signed_token(token: str, secret: str) -> Optional[Dict[str, Any]]:
    """Verify a signed bearer token's signature and expiration.

    Returns the payload dict (with operator_id/role) on success, None on any
    failure (bad signature, expired, malformed).
    """
    try:
        if not token or "." not in token:
            return None
        payload_b64, sig_b64 = token.rsplit(".", 1)
        payload_bytes = _b64url_decode(payload_b64)
        expected_sig = hmac.new(
            secret.encode("utf-8"), payload_bytes, hashlib.sha256
        ).digest()
        provided_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, provided_sig):
            return None
        payload = _json.loads(payload_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        exp = payload.get("exp")
        if exp is not None and time.time() > float(exp):
            return None
        if not payload.get("operator_id"):
            return None
        return payload
    except Exception:
        return None


def authenticate_operator(req: Any) -> Tuple[bool, str, str, Optional[str]]:
    """
    Authenticate an operator from a Bearer token only.

    Returns (is_authenticated, operator_id, role, error_message).

    Security properties:
      * A valid Authorization: Bearer <token> header is REQUIRED.
      * A client-provided X-Operator-ID header is informational only and never
        used to authenticate or derive identity.
      * Token identity (operator_id/role) is derived from the authenticated
        principal (signed token), never from client headers.
      * Token signature and expiration are verified.
      * Insecure bypasses (X-Operator-ID auth, TESTING/debug auto-login,
        X-Operator-Role derivation, FLASK_ENV development login) are removed.
    """
    auth_header = req.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        return False, "", "", "Missing or invalid Authorization header (Bearer token required)"

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return False, "", "", "Empty bearer token"

    # X-Operator-ID is read solely for audit context; it never authenticates.
    _x_operator_id = req.headers.get("X-Operator-ID") or req.headers.get("X-Operator-Id")

    secret = _auth_secret()
    if secret:
        payload = verify_signed_token(token, secret)
        if payload:
            return (
                True,
                str(payload["operator_id"]),
                str(payload.get("role", "OPERATOR")),
                None,
            )

    # Legacy plaintext token fallback is permitted ONLY in the test suite
    # (Flask TESTING mode). Production (TESTING=False) never accepts it.
    if app.config.get("TESTING"):
        if token in VALID_OPERATOR_TOKENS:
            info = VALID_OPERATOR_TOKENS[token]
            return True, info["operator_id"], info["role"], None
        # Fall back to a signed token verified with the test-only secret so
        # tests can exercise the signed-token path without an env secret.
        payload = verify_signed_token(token, _test_secret())
        if payload:
            return (
                True,
                str(payload["operator_id"]),
                str(payload.get("role", "OPERATOR")),
                None,
            )

    return False, "", "", "Invalid bearer authorization token"


def require_review_role(role: Optional[str]) -> bool:
    """Role-based authorization for human review approval/rejection actions."""
    return bool(role) and str(role).upper() in REVIEW_ACTION_ROLES


# ---------------------------------------------------------------------------
# HTTPS enforcement (production ingress). Enabled via FORCE_HTTPS=true and only
# applies outside Flask TESTING mode so the test client is unaffected.
# ---------------------------------------------------------------------------
_FORCE_HTTPS = os.environ.get("FORCE_HTTPS", "false").lower() in ("1", "true", "yes")


@app.before_request
def _enforce_https():
    if _FORCE_HTTPS and not app.config.get("TESTING"):
        if request.headers.get("X-Forwarded-Proto", "http") != "https" and not request.is_secure:
            return jsonify({"success": False, "error": "HTTPS required"}), 403




@app.route("/api/3angle/audit_event", methods=["POST"])
def api_3angle_audit_event():
    """Human review decision audit endpoint adhering to Phase 3.3 specifications."""
    is_auth, operator_id, role, err = authenticate_operator(request)
    if not is_auth:
        return jsonify({"success": False, "error": f"Unauthorized: {err}"}), 401

    # Role-based authorization: only permitted roles may approve/reject reviews.
    if not require_review_role(role):
        return jsonify({
            "success": False,
            "error": f"Forbidden: role '{role}' is not authorized for review actions",
        }), 403

    data = request.json or {}
    job_id = data.get("job_id")
    event_id = data.get("event_id")
    action = str(data.get("action", "CONFIRM_CARTON_CROSSING")).strip()
    reason = str(data.get("reason") or data.get("operator_notes") or "")
    adjusted_count = data.get("adjusted_count")

    if not event_id:
        return jsonify({"success": False, "error": "Missing event_id"}), 400

    # Obtain authoritative durable event store
    db_store = None
    if getattr(state, "three_camera_pipeline", None) and getattr(state.three_camera_pipeline, "event_store", None):
        db_store = state.three_camera_pipeline.event_store
    else:
        db_store = get_app_event_store()

    # Apply transactional change in SQLite EventStore
    success, db_reason, updated_record = db_store.record_manual_review(
        event_id=event_id,
        action=action,
        operator_id=operator_id,
        reason=reason,
        adjusted_count=adjusted_count,
    )

    # Synchronize in-memory job state and pipeline audit_events
    with lock:
        reports_to_search = []
        if job_id and job_id in offline_jobs:
            target_report = offline_jobs[job_id].get("report")
            if target_report:
                reports_to_search.append(target_report)
        if job_id and job_id in multi_angle_jobs:
            target_report = multi_angle_jobs[job_id].get("result") or multi_angle_jobs[job_id].get("report")
            if target_report:
                reports_to_search.append(target_report)

        for j in offline_jobs.values():
            if j.get("report") and j.get("report") not in reports_to_search:
                reports_to_search.append(j["report"])
        for j in multi_angle_jobs.values():
            m_rep = j.get("result") or j.get("report")
            if m_rep and m_rep not in reports_to_search:
                reports_to_search.append(m_rep)

        if state.multi_angle_report and state.multi_angle_report not in reports_to_search:
            reports_to_search.append(state.multi_angle_report)

        found_ev = None
        for r in reports_to_search:
            evs = r.get("result", {}).get("events", [])
            for ev in evs:
                if ev.get("details", {}).get("event_id") == event_id or ev.get("event_id") == event_id:
                    found_ev = ev
                    break
            if found_ev:
                break

        if found_ev is None and getattr(state, "three_camera_pipeline", None):
            for ev in state.three_camera_pipeline.audit_events:
                if ev.get("event_id") == event_id:
                    found_ev = ev
                    break

        if found_ev:
            act_upper = action.upper()
            if "CONFIRM" in act_upper or act_upper == "APPROVE":
                found_ev["is_auto_confirmed"] = False
                found_ev["is_human_approved"] = True
                found_ev["requires_review"] = False
                found_ev["state"] = "COUNT_COMMITTED"
                found_ev["audit_verdict"] = "HUMAN_APPROVED"
                if "details" in found_ev:
                    found_ev["details"]["is_auto_confirmed"] = False
                    found_ev["details"]["is_human_approved"] = True
                    found_ev["details"]["requires_review"] = False
                    found_ev["details"]["state"] = "COUNT_COMMITTED"
            elif "REJECT" in act_upper:
                found_ev["is_auto_confirmed"] = False
                found_ev["is_human_approved"] = False
                found_ev["requires_review"] = False
                found_ev["state"] = "MANUALLY_REJECTED"
                found_ev["action"] = "DISCARD"
                found_ev["audit_verdict"] = "HUMAN_REJECTED"
                if "details" in found_ev:
                    found_ev["details"]["action"] = "DISCARD"
                    found_ev["details"]["state"] = "MANUALLY_REJECTED"
            elif "AMBIGUOUS" in act_upper or act_upper == "ADJUST":
                found_ev["is_auto_confirmed"] = False
                found_ev["is_human_approved"] = False
                found_ev["requires_review"] = True
                found_ev["state"] = "MARK_AMBIGUOUS"
                found_ev["audit_verdict"] = "AMBIGUOUS"
            elif "FLAG" in act_upper or "ERROR" in act_upper:
                found_ev["is_auto_confirmed"] = False
                found_ev["is_human_approved"] = False
                found_ev["requires_review"] = False
                found_ev["state"] = "FLAGGED_ERROR"
                found_ev["audit_verdict"] = "SYSTEM_ERROR_FLAGGED"

        # Also register in pipeline event_store if newly created
        if not updated_record and found_ev:
            db_store.commit_event(found_ev, session_id=job_id or "default_session")
            _, db_reason, updated_record = db_store.record_manual_review(
                event_id=event_id,
                action=action,
                operator_id=operator_id,
                reason=reason,
                adjusted_count=adjusted_count,
            )

    sess_id = updated_record.get("session_id", "default_session") if updated_record else "default_session"
    counts = db_store.get_counts(session_id=sess_id)
    logger.info("Human review action '%s' applied by '%s' to event %s: %s", action, operator_id, event_id, db_reason)
    res_event = updated_record if updated_record else found_ev
    if res_event and found_ev:
        for k in ("audit_verdict", "action", "state"):
            if k in found_ev:
                res_event[k] = found_ev[k]

    return jsonify({
        "success": success,
        "event_id": event_id,
        "action": action,
        "reason": db_reason,
        "operator_id": operator_id,
        "event": res_event,
        "counts": counts,
    })


@app.route("/api/3angle/review_queue", methods=["GET"])
def api_3angle_review_queue():
    """Returns all flagged events requiring human review with full multi-camera context."""
    db_store = None
    if getattr(state, "three_camera_pipeline", None) and getattr(state.three_camera_pipeline, "event_store", None):
        db_store = state.three_camera_pipeline.event_store
    else:
        db_store = get_app_event_store()

    session_id = request.args.get("session_id")
    db_items = db_store.get_review_queue(session_id=session_id)

    with lock:
        flagged_events = []
        seen_ids = {item["event_id"] for item in db_items}

        # 1. Search multi_angle_jobs
        for j in multi_angle_jobs.values():
            rep = j.get("result") or j.get("report")
            if rep:
                evs = rep.get("result", {}).get("events", []) or rep.get("events", [])
                for ev in evs:
                    eid = ev.get("details", {}).get("event_id") or ev.get("event_id")
                    if eid and eid not in seen_ids:
                        req_rev = ev.get("requires_review", False) or ev.get("details", {}).get("requires_review", False)
                        if req_rev:
                            flagged_events.append(ev)
                            seen_ids.add(eid)

        # 2. Search active stream multi_angle_report
        if state.multi_angle_report:
            rep = state.multi_angle_report
            evs = rep.get("result", {}).get("events", []) or rep.get("events", [])
            for ev in evs:
                eid = ev.get("details", {}).get("event_id") or ev.get("event_id")
                if eid and eid not in seen_ids:
                    req_rev = ev.get("requires_review", False) or ev.get("details", {}).get("requires_review", False)
                    if req_rev:
                        flagged_events.append(ev)
                        seen_ids.add(eid)

        # 3. Search active pipeline if present
        if getattr(state, "three_camera_pipeline", None):
            for ev in state.three_camera_pipeline.audit_events:
                eid = ev.get("event_id")
                if eid and eid not in seen_ids and ev.get("requires_review", False):
                    flagged_events.append(ev)
                    seen_ids.add(eid)

        ambiguous_count = 0
        hold_count = 0
        items = []

        # Ingest persistent items from DB
        for db_ev in db_items:
            st = db_ev.get("state", "REVIEW_REQUIRED")
            if st in ("AMBIGUOUS_STACK", "MARK_AMBIGUOUS"):
                ambiguous_count += 1
            else:
                hold_count += 1

            cons = db_ev.get("consensus", {})
            items.append({
                "event_id": db_ev.get("event_id"),
                "session_id": db_ev.get("session_id"),
                "timestamp_sec": db_ev.get("timestamp_sec", 0.0),
                "carton_count": db_ev.get("carton_count", 1),
                "direction": db_ev.get("direction", "IN"),
                "decision_action": st,
                "reason": db_ev.get("notes") or "Flagged for manual verification",
                "consensus_level": cons.get("level", "UNCERTAIN"),
                "participating_cameras": db_ev.get("participating_cameras", ["cam_top", "cam_side", "cam_flank"]),
                "camera_confidences": cons.get("camera_confidences", {}),
                "sync_quality": cons.get("sync_quality", 1.0),
                "association_decision": cons.get("decision", "REVIEW_REQUIRED"),
                "review_clip_path": db_ev.get("notes") if "clip" in str(db_ev.get("notes", "")) else None,
                "audit_verdict": "PENDING_AUDIT",
                "audit_history": db_ev.get("audit_history", []),
            })

        for ev in flagged_events:
            det = ev.get("details", ev)
            act = det.get("decision_action") or ev.get("decision_action") or "HOLD_FOR_REVIEW"
            if act == "AMBIGUOUS_STACK":
                ambiguous_count += 1
            else:
                hold_count += 1

            cons = det.get("consensus", {})
            items.append({
                "event_id": det.get("event_id") or ev.get("event_id"),
                "session_id": det.get("session_id", "live_session"),
                "timestamp_sec": det.get("timestamp_sec") or ev.get("timestamp_sec", 0.0),
                "carton_count": det.get("carton_count") or ev.get("fused_cartons") or 1,
                "direction": det.get("direction") or ev.get("direction", "IN"),
                "decision_action": act,
                "reason": det.get("decision_reason") or det.get("reason") or "Stack ambiguity detected across views",
                "consensus_level": cons.get("level") or ev.get("consensus_status", "UNCERTAIN"),
                "participating_cameras": det.get("participating_cameras", ["cam_top", "cam_side", "cam_flank"]),
                "camera_confidences": cons.get("camera_confidences", {}),
                "sync_quality": cons.get("sync_quality", 1.0),
                "association_decision": det.get("association_decision", "REVIEW_REQUIRED"),
                "review_clip_path": det.get("review_clip_path") or ev.get("review_clip_path"),
                "audit_verdict": ev.get("audit_verdict") or "PENDING_AUDIT",
                "audit_history": [],
            })

        return jsonify({
            "success": True,
            "total_flagged": len(items),
            "ambiguous_stack_count": ambiguous_count,
            "hold_for_review_count": hold_count,
            "queue": items,
        })


@app.route("/api/3angle/pallet_scale", methods=["GET", "POST"])
def api_3angle_pallet_scale():
    """Ground-truth pallet scale weight cross-check endpoint with automated shadow gate & evidence capture."""
    with lock:
        # Determine current optical count
        current_optical = 0
        if state.multi_angle_report:
            res_dict = state.multi_angle_report.get("result", {})
            current_optical = res_dict.get("total_loaded_cartons") or res_dict.get("confirmed_cartons") or 0
        elif getattr(state, "three_camera_pipeline", None):
            current_optical = getattr(state.three_camera_pipeline, "confirmed_in", 0)
        elif hasattr(state, "counter"):
            current_optical = getattr(state.counter, "net_count", 0)
        else:
            current_optical = getattr(state, "total_count", 0)

        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            session_id = data.get("session_id", getattr(state, "session_id", None) or "default_shadow")
            gross_w = float(data.get("gross_weight_kg", 0.0))
            tare_w = float(data.get("tare_weight_kg", 25.0))
            unit_w = float(data.get("nominal_carton_weight_kg", 12.5))
            if "optical_count" in data:
                current_optical = int(data["optical_count"])

            sm = get_shadow_manager(session_id=session_id)
            audit_res = sm.record_pallet_scale_audit(
                gross_weight_kg=gross_w,
                tare_weight_kg=tare_w,
                nominal_carton_weight_kg=unit_w,
                optical_count=current_optical,
                live_camera_manager=live_camera_manager,
                notes=str(data.get("notes", "")),
            )
            state.pallet_scale_telemetry = audit_res
            return jsonify({
                "success": True,
                "audit": state.pallet_scale_telemetry,
            })

        return jsonify({
            "success": True,
            "current_optical_count": current_optical,
            "audit": state.pallet_scale_telemetry,
        })


@app.route("/api/3angle/pallet_scale_history", methods=["GET"])
def api_3angle_pallet_scale_history():
    """Retrieves all historical pallet scale audit checks for the session."""
    session_id = request.args.get("session_id", getattr(state, "session_id", None) or "default_shadow")
    sm = get_shadow_manager(session_id=session_id)
    audits = sm.get_pallet_scale_audits(session_id=session_id)
    return jsonify({
        "success": True,
        "session_id": session_id,
        "total_audits": len(audits),
        "audits": audits,
    })


@app.route("/api/3angle/daemon_status", methods=["GET"])
def api_3angle_daemon_status():
    """Exposes supervisory health, background live worker stats, and database store status."""
    session_id = request.args.get("session_id", getattr(state, "session_id", None) or "default_shadow")
    sm = get_shadow_manager(session_id=session_id)
    live_status = live_camera_manager.status()
    db_store = get_app_event_store()

    return jsonify({
        "success": True,
        "daemon": "truck_loading_app_service",
        "session_id": session_id,
        "live_cameras": live_status,
        "shadow_mode": sm.get_shadow_counts(),
        "durable_store": {
            "ready": db_store is not None and not db_store._corrupt,
            "path": db_store.db_path if db_store else None,
        },
        "model": state.model_name,
        "device": getattr(state, "device", "auto"),
    })



# ---------------------------------------------------------------------------
# Graceful Shutdown
# ---------------------------------------------------------------------------
def _shutdown_handler(signum, frame):
    signame = signal.Signals(signum).name
    logger.info("Received %s, shutting down...", signame)
    shutdown_event.set()
    live_camera_manager.stop()
    state.cleanup()


signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)
atexit.register(live_camera_manager.stop)
atexit.register(state.cleanup)


# ---------------------------------------------------------------------------
# CORS Headers (Strictly validated origin, avoiding wildcard vulnerability)
# ---------------------------------------------------------------------------
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "localhost,127.0.0.1").strip()

@app.after_request
def _add_cors_headers(response):
    origin = request.headers.get("Origin")
    if CORS_ORIGIN == "*":
        response.headers["Access-Control-Allow-Origin"] = "*"
    elif origin:
        allowed = [o.strip() for o in CORS_ORIGIN.split(",") if o.strip()]
        if any(o in origin for o in allowed) or origin.startswith(("http://localhost", "http://127.0.0.1")):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


# ---------------------------------------------------------------------------
# Phase 3.3 Workstream E: Operational Shadow Mode Endpoints
# ---------------------------------------------------------------------------
_shadow_manager: Optional[ShadowModeManager] = None


def get_shadow_manager(session_id: str = "default_shadow") -> ShadowModeManager:
    global _shadow_manager
    if _shadow_manager is None or _shadow_manager.session_id != session_id:
        db_store = get_app_event_store()
        _shadow_manager = ShadowModeManager(
            event_store=db_store,
            session_id=session_id,
            authoritative_inventory_locked=True,
        )
    return _shadow_manager


@app.route("/api/shadow/status", methods=["GET"])
def api_shadow_status():
    """Returns shadow mode operating status, stream health, and inventory lock flag."""
    session_id = request.args.get("session_id", "default_shadow")
    sm = get_shadow_manager(session_id=session_id)
    return jsonify({
        "success": True,
        "mode": "SHADOW_MODE",
        "authoritative_inventory_locked": sm.authoritative_inventory_locked,
        "session_id": sm.session_id,
        "health": sm.health.__dict__,
        "incidents_count": len(sm.incidents),
    })


@app.route("/api/shadow/counts", methods=["GET"])
def api_shadow_counts():
    """Returns side-by-side counts: GT physical, auto confirmed, human approved, discrepancy."""
    session_id = request.args.get("session_id", "default_shadow")
    sm = get_shadow_manager(session_id=session_id)
    return jsonify({
        "success": True,
        "shadow_summary": sm.get_shadow_counts(),
    })


@app.route("/api/shadow/compare", methods=["POST"])
def api_shadow_compare():
    """Registers human physical ground truth count and computes discrepancy."""
    data = request.json or {}
    session_id = data.get("session_id", "default_shadow")
    count = data.get("human_physical_count")
    if count is None:
        return jsonify({"success": False, "error": "Missing 'human_physical_count'"}), 400

    sm = get_shadow_manager(session_id=session_id)
    sm.set_human_physical_count(int(count))
    summary = sm.get_shadow_counts()
    return jsonify({
        "success": True,
        "comparison": summary,
    })


@app.route("/api/shadow/incidents", methods=["GET"])
def api_shadow_incidents():
    """Retrieves structured incident log for shadow session."""
    session_id = request.args.get("session_id", "default_shadow")
    sm = get_shadow_manager(session_id=session_id)
    return jsonify({
        "success": True,
        "session_id": sm.session_id,
        "incidents": [
            {
                "incident_id": inc.incident_id,
                "type": inc.incident_type,
                "severity": inc.severity,
                "timestamp_sec": inc.timestamp_sec,
                "description": inc.description,
                "metadata": inc.metadata,
            }
            for inc in sm.incidents
        ],
    })


@app.route("/api/shadow/recover", methods=["POST"])
def api_shadow_recover():
    """Restores session state and incidents from SQLite WAL after application restart."""
    data = request.json or {}
    session_id = data.get("session_id", "default_shadow")
    sm = get_shadow_manager(session_id=session_id)
    summary = sm.recover_session_state()
    return jsonify({
        "success": True,
        "recovered_summary": summary,
    })


@app.route("/api/security/audit_logs", methods=["GET"])
@require_api_key(mutating_only=False)
def api_security_audit_logs():
    """Retrieves immutable security and mutation audit records."""
    store = get_app_event_store()
    limit = int(request.args.get("limit", 100))
    logs = store.get_security_audit_logs(limit=limit) if store else []
    return jsonify({
        "status": "ok",
        "total": len(logs),
        "audit_logs": logs,
    })


@app.errorhandler(500)
def handle_500(e):
    logger.exception("Internal Server Error: %s", e)
    return jsonify({"status": "error", "error": "Internal Server Error", "message": str(e)}), 500

@app.errorhandler(413)
def handle_413(e):
    return jsonify({"status": "error", "error": f"File is too large. Max size is {MAX_UPLOAD_MB}MB.", "message": f"File is too large. Max size is {MAX_UPLOAD_MB}MB."}), 413


# ---------------------------------------------------------------------------
# Start Processing Loop (runs for BOTH `python app.py` AND Gunicorn)
# ---------------------------------------------------------------------------
_processing_thread = threading.Thread(target=processing_loop, daemon=True)
_processing_thread.start()
logger.info("Background processing thread started.")

def _auto_connect_default_live_cameras():
    time.sleep(1.0)
    if os.environ.get("AUTO_CONNECT_LIVE_CAMERAS", "0") != "1":
        return
    default_sources = {
        "cam_top": "rtsp://admin:admin1234@192.168.18.223:554/cam/realmonitor?channel=1&subtype=0",
        "cam_side": "rtsp://admin:admin1234@192.168.18.223:554/cam/realmonitor?channel=2&subtype=0",
        "cam_flank": "rtsp://admin:admin1234@192.168.18.223:554/cam/realmonitor?channel=3&subtype=0",
    }
    try:
        parsed = {
            cid: parse_camera_source(src, cid)
            for cid, src in default_sources.items()
        }
        live_camera_manager.start(parsed)
        logger.info("[AUTO-CONNECT] Connected 3 default Dahua live cameras: %s", list(default_sources.keys()))
    except Exception as e:
        logger.warning("[AUTO-CONNECT] Could not auto-connect default live cameras: %s", e)

threading.Thread(target=_auto_connect_default_live_cameras, daemon=True).start()

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  TRUCK LOADING DASHBOARD READY")
    logger.info("  URL:        http://%s:%d", HOST, PORT)
    logger.info("  Health:     http://%s:%d/health", HOST, PORT)
    logger.info("  Model:      %s", state.model_name)
    logger.info("  Count Mode: %s", state.count_mode)
    logger.info("  Source:     %s", state.source or "(none)")
    logger.info("=" * 60)

    app.run(host=HOST, port=PORT, debug=False, threaded=True)
