import os
import sys
import time
import threading
import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from counter_engine import TripwireCounter
from visualizer import LoadingVisualizer

# Enable CPU multi-threading
torch.set_num_threads(4)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# State lock and shared pipeline state
lock = threading.Lock()

class StreamState:
    def __init__(self):
        self.source = "sample_truck_loading.mp4"
        self.model_name = "yolov8n.pt"
        self.model = YOLO(self.model_name)
        self.line_x = 515
        self.conf = 0.30
        self.iou = 0.50
        self.hysteresis = 15
        self.cooldown = 15
        self.frame_stride = 2  # Run heavy YOLO detection every 2nd frame for 30 FPS real-time speed
        self.img_size = 384    # Optimized inference resolution
        self.target_classes = ["person", "carton", "box", "suitcase", "backpack"]
        
        self.counter = TripwireCounter(self.line_x, self.hysteresis, cooldown_frames=self.cooldown)
        self.visualizer = LoadingVisualizer()
        
        self.is_paused = False
        self.loop_video = True
        self.fps = 25.0
        self.live_fps = 25.0
        self.frame_idx = 0
        self.active_count = 0
        self.video_width = 1024
        self.video_height = 576
        
        self.cap = None
        self.current_frame_jpg = None
        self.last_tracked_objects = []
        self.init_capture()

    def init_capture(self):
        if self.cap is not None:
            self.cap.release()
        
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.cap = cv2.VideoCapture(src)
        if self.cap.isOpened():
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1024
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 576
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            if self.line_x is None or self.line_x > self.video_width:
                self.line_x = self.video_width // 2
                self.counter.set_line_x(self.line_x)
        self.frame_idx = 0
        self.last_tracked_objects = []

state = StreamState()


def processing_loop():
    """High-performance background worker with frame-stride optimization for 30 FPS real-time streaming."""
    global state
    
    model_names = state.model.names
    desired = set(state.target_classes)
    target_cids = [cid for cid, name in model_names.items() if any(d in name.lower() for d in desired)]
    if not target_cids:
        target_cids = None

    while True:
        if state.is_paused:
            time.sleep(0.04)
            continue

        with lock:
            if state.cap is None or not state.cap.isOpened():
                state.init_capture()
                time.sleep(0.05)
                continue

            t_start = time.time()
            ret, frame = state.cap.read()
            if not ret:
                if state.loop_video and not isinstance(state.source, int):
                    state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    state.frame_idx = 0
                    continue
                else:
                    time.sleep(0.05)
                    continue

            state.frame_idx += 1
            
            # Fast YOLO Tracking on frame stride
            run_detection = (state.frame_idx % state.frame_stride == 0) or (len(state.last_tracked_objects) == 0)
            
            if run_detection:
                results = state.model.track(
                    frame,
                    persist=True,
                    classes=target_cids,
                    conf=state.conf,
                    iou=state.iou,
                    imgsz=state.img_size,
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

                state.last_tracked_objects = tracked_objects
            else:
                tracked_objects = state.last_tracked_objects

            state.active_count = len(tracked_objects)

            # Update directional counter
            if run_detection:
                state.counter.update(tracked_objects, state.frame_idx)

            # Visualizer renders overlays
            state.visualizer.draw_trajectories(frame, state.counter.track_history)
            state.visualizer.draw_tripwire(frame, state.line_x)
            state.visualizer.draw_detections(frame, tracked_objects, state.counter.track_side, state.line_x)
            state.visualizer.draw_hud(
                frame=frame,
                total_in=state.counter.total_in,
                total_out=state.counter.total_out,
                net_count=state.counter.net_count,
                fps=state.live_fps,
                active_count=state.active_count,
                recent_event=state.counter.recent_event,
                recent_event_expiry=state.counter.recent_event_expiry,
            )

            # Efficient JPEG encoding (quality 75 for speed and low bandwidth)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            state.current_frame_jpg = buffer.tobytes()

            t_end = time.time()
            frame_dt = max(t_end - t_start, 1e-4)
            state.live_fps = 0.9 * state.live_fps + 0.1 * (1.0 / frame_dt)

        # Rate limiter to maintain smooth natural playback speed
        target_dt = 1.0 / max(state.fps, 10.0)
        sleep_dt = target_dt - (time.time() - t_start)
        if sleep_dt > 0:
            time.sleep(sleep_dt)


def generate_frames():
    """MJPEG stream generator."""
    while True:
        if state.current_frame_jpg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + state.current_frame_jpg + b'\r\n')
        time.sleep(0.03)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def api_stats():
    with lock:
        recent_events = list(state.counter.events[-10:])
        recent_events.reverse()  # Newest first
        return jsonify({
            "total_in": state.counter.total_in,
            "total_out": state.counter.total_out,
            "net_count": state.counter.net_count,
            "fps": round(state.live_fps, 1),
            "active_count": state.active_count,
            "frame_idx": state.frame_idx,
            "line_x": state.line_x,
            "video_width": state.video_width,
            "video_height": state.video_height,
            "is_paused": state.is_paused,
            "loop_video": state.loop_video,
            "source": state.source,
            "events": recent_events,
        })


@app.route('/api/set_line_x', methods=['POST'])
def api_set_line_x():
    data = request.json or {}
    new_x = int(data.get('line_x', state.line_x))
    with lock:
        state.line_x = max(10, min(state.video_width - 10, new_x))
        state.counter.set_line_x(state.line_x)
    return jsonify({"success": True, "line_x": state.line_x})


@app.route('/api/toggle_pause', methods=['POST'])
def api_toggle_pause():
    with lock:
        state.is_paused = not state.is_paused
    return jsonify({"success": True, "is_paused": state.is_paused})


@app.route('/api/toggle_loop', methods=['POST'])
def api_toggle_loop():
    with lock:
        state.loop_video = not state.loop_video
    return jsonify({"success": True, "loop_video": state.loop_video})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    with lock:
        state.counter.reset_counts()
    return jsonify({"success": True})


@app.route('/api/set_source', methods=['POST'])
def api_set_source():
    data = request.json or {}
    src = data.get('source', 'sample_truck_loading.mp4')
    with lock:
        state.source = src
        state.counter.reset_counts()
        state.init_capture()
    return jsonify({"success": True, "source": state.source})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'video' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    with lock:
        state.source = filepath
        state.counter.reset_counts()
        state.init_capture()

    return jsonify({"success": True, "filename": filename, "source": filepath})


@app.route('/api/update_params', methods=['POST'])
def api_update_params():
    data = request.json or {}
    with lock:
        if 'conf' in data:
            state.conf = float(data['conf'])
        if 'iou' in data:
            state.iou = float(data['iou'])
        if 'hysteresis' in data:
            state.hysteresis = int(data['hysteresis'])
            state.counter.hysteresis = state.hysteresis
    return jsonify({"success": True, "conf": state.conf, "iou": state.iou, "hysteresis": state.hysteresis})


if __name__ == '__main__':
    # Start background processing thread
    thread = threading.Thread(target=processing_loop, daemon=True)
    thread.start()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"============================================================")
    print(f"  TRUCK LOADING WEB DASHBOARD READY")
    print(f"  Live Stream: http://localhost:{port}")
    print(f"============================================================")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
