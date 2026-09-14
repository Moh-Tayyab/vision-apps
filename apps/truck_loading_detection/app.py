import os
import sys
import time
import signal
import logging
import threading
import traceback
import atexit
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
import json
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from counter_engine import TripwireCounter
from visualizer import LoadingVisualizer
from process_video_pipeline import process_video_file

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
CONF = float(os.environ.get("CONF_THRESHOLD", 0.24))
IOU = float(os.environ.get("IOU_THRESHOLD", 0.45))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 416))
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", 2))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 500))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", 70))
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# Default video source (RTSP URL or file path) via env
DEFAULT_SOURCE = os.environ.get("VIDEO_SOURCE", "")

torch.set_num_threads(TORCH_THREADS)

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

lock = threading.Lock()
shutdown_event = threading.Event()

# ---------------------------------------------------------------------------
# Counting Presets
# ---------------------------------------------------------------------------
COUNT_MODES = {
    "cartons_only": {"carton", "box", "cardboard box", "package", "suitcase", "backpack"},
    "people_only": {"person"},
    "all": None,
}

VIDEO_PRESETS = {
    "sample_truck_loading.mp4": {"line_x_ratio": 0.50, "direction": "left_to_right"},
    "warehouse_1.mp4": {"line_x_ratio": 0.50, "direction": "right_to_left"},
    "warehouse_2.mp4": {"line_x_ratio": 0.65, "direction": "right_to_left"},
    "warehouse_3.mp4": {"line_x_ratio": 0.50, "direction": "right_to_left"},
    "warehouse_4.mp4": {"line_x_ratio": 0.40, "direction": "left_to_right"},
    "warehouse_5.mp4": {"line_x_ratio": 0.45, "direction": "right_to_left"},
    "warehouse_6.mp4": {"line_x_ratio": 0.45, "direction": "right_to_left"},
    "warehouse_7.mp4": {"line_x_ratio": 0.22, "direction": "right_to_left"},
    "warehouse_8.mp4": {"line_x_ratio": 0.65, "direction": "left_to_right"},
}


# ---------------------------------------------------------------------------
# Stream State
# ---------------------------------------------------------------------------
class StreamState:
    def __init__(self):
        app_dir = os.path.dirname(os.path.abspath(__file__))
        cand_world = os.path.abspath(os.path.join(app_dir, "..", "..", "yolov8s-worldv2.pt"))
        if os.path.exists(cand_world):
            self.model_name = cand_world
        else:
            self.model_name = os.path.join(app_dir, "yolov8n.pt")
            if not os.path.exists(self.model_name):
                self.model_name = os.path.join(app_dir, "yolo11n.pt")
            if not os.path.exists(self.model_name):
                self.model_name = "yolov8n.pt"

        logger.info("Loading YOLO model: %s", self.model_name)
        self.model = YOLO(self.model_name)
        if hasattr(self.model, "set_classes"):
            try:
                self.model.set_classes(["person", "cardboard box", "carton"])
            except Exception:
                pass

        self.source = DEFAULT_SOURCE
        self.line_x = 515
        self.loading_direction = "left_to_right"
        self.conf = CONF
        self.iou = IOU
        self.hysteresis = 15
        self.cooldown = 15
        self.frame_stride = FRAME_STRIDE
        self.img_size = IMG_SIZE
        self.target_classes = ["person", "carton", "box", "cardboard box", "package", "suitcase", "backpack"]
        self.count_mode = "cartons_only"

        self.counter = TripwireCounter(
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

        self.is_paused = False
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

        try:
            # Set RTSP transport timeout for live camera streams
            if isinstance(src, str) and src.startswith("rtsp://"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

            self.cap = cv2.VideoCapture(src)
            if self.cap.isOpened():
                raw_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1024
                raw_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 576
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
                if self.fps <= 0 or self.fps > 120:
                    self.fps = 25.0

                max_dim = max(raw_w, raw_h)
                if max_dim > 720:
                    scale = 720.0 / max_dim
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
            if state.is_paused:
                time.sleep(0.04)
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
                    backoff = min(10.0, 1.0 + (state.reconnect_attempts * 1.5))
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
                    is_live_stream = isinstance(source, int) or (
                        isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("http"))
                    )
                    if is_live_stream:
                        logger.warning("Live stream frame drop. Reconnecting...")
                        state.connection_status = "reconnecting"
                        if state.cap:
                            state.cap.release()
                        state.cap = None
                        state.last_reconnect_time = time.time()
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

            if scale_fac != 1.0:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            run_detection = (cur_frame_idx % stride_val == 0) or (len(last_objs) == 0)

            if run_detection:
                results = state.model.track(
                    frame,
                    persist=True,
                    classes=target_cids,
                    conf=conf_val,
                    iou=iou_val,
                    imgsz=img_size_val,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )

                tracked_objects = []
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()

                    for box, tid, c_idx, c_val in zip(boxes, track_ids, classes, confs):
                        c_name = model_names.get(int(c_idx), f"class_{int(c_idx)}")
                        tracked_objects.append((int(tid), tuple(box), c_name, float(c_val)))
            else:
                tracked_objects = last_objs

            with lock:
                state.last_tracked_objects = tracked_objects
                state.active_count = len(tracked_objects)

                if run_detection:
                    state.counter.update(tracked_objects, cur_frame_idx, frame=frame)

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
                    associations=state.counter.current_associations,
                    carton_boxes=state.counter.track_carton_boxes,
                    holding_scores=state.counter.track_holding_scores,
                    worker_qty=state.counter.current_worker_qty,
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
                    worker_trips=state.counter.worker_trips_in,
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
    return render_template("index.html")


@app.route("/health")
def health_check():
    uptime = (datetime.now(timezone.utc) - state.start_time).total_seconds()
    return jsonify({
        "status": "healthy" if state.connection_status != "error" else "degraded",
        "service": "truck-loading-detection",
        "connection_status": state.connection_status,
        "source": state.source or None,
        "model": state.model_name,
        "uptime_seconds": round(uptime, 1),
        "net_count": state.counter.net_count,
        "fps": round(state.live_fps, 1),
    })


@app.route("/video_feed")
def video_feed():
    resp = Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.route("/api/stats")
def api_stats():
    with lock:
        recent_events = list(state.counter.events)
        recent_events.reverse()
        recent_events = recent_events[:15]
        return jsonify({
            "total_in": state.counter.total_in,
            "total_out": state.counter.total_out,
            "net_count": state.counter.net_count,
            "worker_trips_in": state.counter.worker_trips_in,
            "worker_trips_out": state.counter.worker_trips_out,
            "loading_direction": state.counter.loading_direction,
            "fps": round(state.live_fps, 1),
            "active_count": state.active_count,
            "frame_idx": state.frame_idx,
            "line_x": state.line_x,
            "video_width": state.video_width,
            "video_height": state.video_height,
            "is_paused": state.is_paused,
            "loop_video": state.loop_video,
            "source": state.source,
            "connection_status": state.connection_status,
            "reconnect_attempts": state.reconnect_attempts,
            "count_mode": state.count_mode,
            "active_associations": state.counter.current_associations,
            "events": recent_events,
            "conf": round(state.conf, 2),
            "hysteresis": state.counter.hysteresis,
        })


@app.route("/api/reset_line_preset", methods=["POST"])
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
def api_set_line_x():
    data = request.json or {}
    new_x = int(data.get("line_x", state.line_x))
    with lock:
        state.line_x = max(10, min(state.video_width - 10, new_x))
        state.counter.set_line_x(state.line_x)
    return jsonify({"success": True, "line_x": state.line_x})


@app.route("/api/toggle_direction", methods=["POST"])
def api_toggle_direction():
    with lock:
        new_dir = "right_to_left" if state.counter.loading_direction == "left_to_right" else "left_to_right"
        state.loading_direction = new_dir
        state.counter.set_loading_direction(new_dir)
    return jsonify({"success": True, "loading_direction": state.loading_direction})


@app.route("/api/set_count_mode", methods=["POST"])
def api_set_count_mode():
    data = request.json or {}
    mode = data.get("mode", "cartons_only")
    if mode not in COUNT_MODES:
        return jsonify({"error": f"Invalid mode. Choose from {list(COUNT_MODES.keys())}"}), 400
    with lock:
        state.count_mode = mode
        state.counter.set_countable_classes(COUNT_MODES[mode])
    return jsonify({"success": True, "count_mode": state.count_mode})


@app.route("/api/toggle_pause", methods=["POST"])
def api_toggle_pause():
    with lock:
        state.is_paused = not state.is_paused
    return jsonify({"success": True, "is_paused": state.is_paused})


@app.route("/api/toggle_loop", methods=["POST"])
def api_toggle_loop():
    with lock:
        state.loop_video = not state.loop_video
    return jsonify({"success": True, "loop_video": state.loop_video})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with lock:
        state.counter.reset_counts()
    return jsonify({"success": True})


@app.route("/api/set_source", methods=["POST"])
def api_set_source():
    data = request.json or {}
    src = data.get("source", "")
    with lock:
        state.source = src
        state.counter.reset_counts()
        state.init_capture()
    logger.info("Source changed to: %s", src or "(none)")
    return jsonify({"success": True, "source": state.source, "connection_status": state.connection_status})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    raw_name = file.filename
    base, ext = os.path.splitext(raw_name)
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Only video files ({', '.join(ALLOWED_EXTENSIONS)}) are allowed"}), 400

    safe_base = secure_filename(base)
    if not safe_base:
        safe_base = f"video_{int(time.time())}"
    filename = f"{safe_base}{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Cleanup old uploads (keep last 10 files)
    try:
        upload_dir = app.config["UPLOAD_FOLDER"]
        files = sorted(
            [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if f.endswith(ALLOWED_EXTENSIONS)],
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
    return jsonify({"success": True, "filename": filename, "source": filepath})


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

            res = process_video_file(
                input_path=inp_path,
                progress_callback=_prog,
            )
            if "error" in res:
                offline_jobs[jid]["status"] = "failed"
                offline_jobs[jid]["error"] = res["error"]
            else:
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
# Graceful Shutdown
# ---------------------------------------------------------------------------
def _shutdown_handler(signum, frame):
    signame = signal.Signals(signum).name
    logger.info("Received %s, shutting down...", signame)
    shutdown_event.set()
    state.cleanup()


signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)
atexit.register(state.cleanup)


# ---------------------------------------------------------------------------
# CORS Headers (for cross-origin browser access)
# ---------------------------------------------------------------------------
@app.after_request
def _add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ---------------------------------------------------------------------------
# Start Processing Loop (runs for BOTH `python app.py` AND Gunicorn)
# ---------------------------------------------------------------------------
_processing_thread = threading.Thread(target=processing_loop, daemon=True)
_processing_thread.start()
logger.info("Background processing thread started.")

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
