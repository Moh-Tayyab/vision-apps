"""FastAPI app: Helmet Detection (App 2).

Live video from a mobile camera is consumed via frame push:
POST /ingest/frame with JPEG frames, then GET /stream/detect for
annotated live output. Includes a live dashboard and mobile camera UI.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from detector import HelmetDetector, PersonStatus, FrameResult, draw_helmet_detections
from streamer import FrameBuffer, mjpeg_from_buffer, apply_transform

load_dotenv(find_dotenv(usecwd=True))

app = FastAPI(
    title="Helmet Detection",
    description="Detect persons with/without helmets from live camera frames",
    version="1.1.0",
)

_detector: Optional[HelmetDetector] = None
_ingest_buffer = FrameBuffer(max_frames=10)
_violations_log: deque = deque(maxlen=200)

_CAMERA_TRANSFORM = os.getenv("CAMERA_TRANSFORM", "none")
_camera_capture_running = False
_camera_capture_thread: Optional[threading.Thread] = None

_last_detection_info = {
    "num_persons": 0,
    "num_helmets": 0,
    "num_violations": 0,
    "persons": [],
    "violations": [],
    "inference_time_ms": 0.0,
    "timestamp": 0.0,
}
_cached_result: Optional[FrameResult] = None
_detection_lock = threading.Lock()
_worker_running = True


def _get_detector() -> HelmetDetector:
    global _detector
    if _detector is None:
        _detector = HelmetDetector()
    return _detector


def _get_local_ip() -> str:
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _read_image(file_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    return img


def _async_detection_worker():
    """Continuous background worker that runs AI inference on fresh video frames."""
    global _cached_result, _last_detection_info
    last_seen_ts = 0.0

    while _worker_running:
        try:
            if not _ingest_buffer.is_active:
                time.sleep(0.04)
                continue

            latest = _ingest_buffer.get_latest()
            if latest is None:
                time.sleep(0.04)
                continue

            ts, frame = latest
            if ts != last_seen_ts:
                last_seen_ts = ts
                conf = float(os.getenv("CONF_THRESHOLD", "0.40"))
                result = _get_detector().detect(frame, confidence=conf)

                now = time.time()
                for v in result.violations:
                    _violations_log.append({"timestamp": now, **v.to_dict()})

                safe_count = sum(1 for p in result.persons if p.status == "helmet")
                violation_count = len(result.violations)

                with _detection_lock:
                    _cached_result = result
                    _last_detection_info = {
                        "num_persons": len(result.persons),
                        "num_helmets": safe_count,
                        "num_violations": violation_count,
                        "persons": [p.to_dict() for p in result.persons],
                        "violations": [v.to_dict() for v in result.violations],
                        "inference_time_ms": result.inference_time_ms,
                        "timestamp": now,
                    }
            else:
                time.sleep(0.02)
        except Exception as e:
            time.sleep(0.05)


_https_started = False


def _start_https_server():
    global _https_started
    if _https_started:
        return
    _https_started = True

    cert_file = os.path.join(os.path.dirname(__file__), "certs", "cert.pem")
    key_file = os.path.join(os.path.dirname(__file__), "certs", "key.pem")
    https_port = int(os.getenv("HTTPS_PORT", "8444"))

    if not (os.path.exists(cert_file) and os.path.exists(key_file)):
        try:
            os.makedirs(os.path.dirname(cert_file), exist_ok=True)
            import subprocess
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_file, "-out", cert_file,
                "-days", "365", "-nodes", "-subj", "/CN=helmet-detection"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    if os.path.exists(cert_file) and os.path.exists(key_file):
        try:
            import uvicorn
            print(f"Starting Helmet Detection HTTPS server on https://0.0.0.0:{https_port}")
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=https_port,
                ssl_keyfile=key_file,
                ssl_certfile=cert_file,
                log_level="warning",
                lifespan="off",
            )
            server = uvicorn.Server(config)
            server.run()
        except Exception as e:
            print(f"HTTPS server error: {e}")


@app.on_event("startup")
async def startup():
    try:
        _get_detector()
    except Exception as e:
        print(f"Warning: detector init failed: {e}")

    t = threading.Thread(target=_async_detection_worker, daemon=True)
    t.start()
    print("Async background AI helmet detection worker started.")

    https_t = threading.Thread(target=_start_https_server, daemon=True)
    https_t.start()


@app.get("/health")
async def health():
    detector = _get_detector()
    return {
        "status": "healthy",
        "backend": detector.backend,
        "ingest_active": _ingest_buffer.is_active,
        "ingest_frames": _ingest_buffer.frame_count,
    }


@app.get("/model/info")
async def model_info():
    return _get_detector().get_model_info()


@app.post("/detect")
async def detect(file: UploadFile = File(...), confidence: Optional[float] = Query(default=None)):
    image = _read_image(await file.read())
    conf = confidence if confidence is not None else float(os.getenv("CONF_THRESHOLD", "0.40"))
    result = _get_detector().detect(image, confidence=conf)
    return result.to_dict()


@app.post("/detect/visualize")
async def detect_visualize(file: UploadFile = File(...), confidence: Optional[float] = Query(default=None)):
    image = _read_image(await file.read())
    conf = confidence if confidence is not None else float(os.getenv("CONF_THRESHOLD", "0.40"))
    result = _get_detector().detect(image, confidence=conf)
    vis = draw_helmet_detections(image, result)
    _, buffer = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/ingest/frame")
async def ingest_frame(file: UploadFile = File(...)):
    """Push one JPEG frame from the mobile camera."""
    image = _read_image(await file.read())
    image = apply_transform(image, _CAMERA_TRANSFORM)
    _ingest_buffer.update(image)

    with _detection_lock:
        info = dict(_last_detection_info)

    return {
        "status": "accepted",
        "ingest_frames": _ingest_buffer.frame_count,
        "num_persons": info.get("num_persons", 0),
        "num_helmets": info.get("num_helmets", 0),
        "num_violations": info.get("num_violations", 0),
        "size": [image.shape[1], image.shape[0]],
    }


@app.get("/violations")
async def violations(limit: int = Query(default=50, ge=1, le=200)):
    """Recent no-helmet events."""
    items = list(_violations_log)[-limit:]
    return {"count": len(items), "violations": items}


@app.get("/usb/cameras")
async def list_usb_cameras():
    """List available USB cameras without opening them (never powers on laptop webcam)."""
    import glob
    devices = []
    for path in sorted(glob.glob("/dev/video*")):
        idx = int(path.replace("/dev/video", ""))
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
    return {"cameras": devices}


def _camera_worker(source: str | int, fps: int = 20):
    global _camera_capture_running
    try:
        src = int(source) if str(source).isdigit() else str(source)
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        delay = 1.0 / max(1, fps)
        while _camera_capture_running:
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = apply_transform(frame, _CAMERA_TRANSFORM)
                _ingest_buffer.update(frame)
            else:
                time.sleep(0.05)
            time.sleep(delay * 0.4)
        cap.release()
    except Exception as e:
        print(f"Camera worker error: {e}")


@app.post("/usb/start")
async def usb_start(
    device_index: int = Query(default=0, description="USB device index (0=/dev/video0, 2=/dev/video2)"),
    fps: int = Query(default=20),
):
    """Start USB webcam capture."""
    global _camera_capture_running, _camera_capture_thread
    _camera_capture_running = False
    if _camera_capture_thread is not None:
        _camera_capture_thread.join(timeout=1.0)
    _camera_capture_running = True
    _camera_capture_thread = threading.Thread(target=_camera_worker, args=(device_index, fps), daemon=True)
    _camera_capture_thread.start()
    return {"status": "started", "device": f"/dev/video{device_index}"}


@app.post("/usb/stop")
async def usb_stop():
    """Stop USB webcam capture."""
    global _camera_capture_running
    _camera_capture_running = False
    return {"status": "stopped"}


@app.post("/camera/connect")
async def camera_connect(
    source_uri: str = Query(..., description="e.g. http://192.168.1.39:8080/video or rtsp://..."),
    fps: int = Query(default=15),
):
    """Connect directly to an IP Webcam or RTSP stream."""
    global _camera_capture_running, _camera_capture_thread
    _camera_capture_running = False
    if _camera_capture_thread is not None:
        _camera_capture_thread.join(timeout=1.0)
    _camera_capture_running = True
    _camera_capture_thread = threading.Thread(target=_camera_worker, args=(source_uri.strip(), fps), daemon=True)
    _camera_capture_thread.start()
    return {"status": "connected", "source": source_uri}


@app.post("/camera/disconnect")
async def camera_disconnect():
    """Disconnect active camera."""
    global _camera_capture_running
    _camera_capture_running = False
    return {"status": "disconnected"}


@app.post("/camera/transform")
async def set_camera_transform(transform: str = Query(default="none")):
    """Live-adjust orientation: none | flip_h | flip_v | rotate_90_cw | rotate_90_ccw | rotate_180."""
    global _CAMERA_TRANSFORM
    valid = ("none", "flip_h", "flip_v", "rotate_90_cw", "rotate_90_ccw", "rotate_180", "rotate_90", "rotate_270")
    t_clean = transform.lower().strip()
    if t_clean not in valid:
        raise HTTPException(status_code=422, detail=f"Invalid transform '{transform}'. Must be one of: {valid}")
    if t_clean in ("rotate_90", "90"):
        t_clean = "rotate_90_cw"
    elif t_clean in ("rotate_270", "270"):
        t_clean = "rotate_90_ccw"
    _CAMERA_TRANSFORM = t_clean
    return {"status": "ok", "transform": _CAMERA_TRANSFORM}


@app.get("/stream")
async def stream():
    gen = mjpeg_from_buffer(_ingest_buffer, fps_limit=30.0)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stream/detect")
async def stream_detect():
    """Live annotated MJPEG: green=helmet, red=no_helmet, orange=unknown."""
    def annotate(frame: np.ndarray) -> np.ndarray:
        with _detection_lock:
            cached = _cached_result
        if cached is not None:
            return draw_helmet_detections(frame, cached)
        return frame

    gen = mjpeg_from_buffer(_ingest_buffer, quality=75, transform=annotate, fps_limit=30.0)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stream/status")
async def stream_status():
    local_ip = _get_local_ip()
    port = int(os.getenv("PORT", "8002"))
    https_port = int(os.getenv("HTTPS_PORT", "8444"))
    with _detection_lock:
        info = dict(_last_detection_info)
    return {
        "ingest_active": _ingest_buffer.is_active,
        "ingest_frames": _ingest_buffer.frame_count,
        "latest_detection": info,
        "transform": _CAMERA_TRANSFORM,
        "local_ip": local_ip,
        "port": port,
        "https_port": https_port,
    }


@app.get("/mobile", response_class=HTMLResponse)
async def mobile_page():
    local_ip = _get_local_ip()
    port = int(os.getenv("PORT", "8002"))
    https_port = int(os.getenv("HTTPS_PORT", "8444"))
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>Helmet Detection - Mobile Camera</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 15px; text-align: center; }}
            .header {{ margin-bottom: 12px; }}
            h1 {{ font-size: 1.4rem; color: #38bdf8; margin-bottom: 4px; }}
            p {{ font-size: 0.85rem; color: #94a3b8; }}
            
            .https-banner {{ background: #7c2d12; border: 1px solid #ea580c; border-radius: 12px; padding: 12px; margin-bottom: 15px; text-align: left; font-size: 0.85rem; }}
            .https-banner h3 {{ color: #fdba74; font-size: 0.95rem; margin-bottom: 4px; }}
            
            .camera-container {{ position: relative; width: 100%; max-width: 460px; margin: 0 auto 12px; border-radius: 16px; overflow: hidden; background: #1e293b; border: 2px solid #334155; aspect-ratio: 4/3; }}
            video {{ width: 100%; height: 100%; object-fit: cover; }}
            canvas {{ display: none; }}
            .snap-preview {{ max-width: 100%; border-radius: 8px; margin-top: 10px; display: none; }}
            
            .stats-badge {{ position: absolute; top: 10px; left: 10px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(6px); border: 1px solid #38bdf8; border-radius: 20px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600; color: #38bdf8; display: flex; align-items: center; gap: 6px; }}
            .live-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #38bdf8; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.4; transform: scale(0.85); }} }}
            
            .controls {{ display: flex; flex-direction: column; gap: 10px; max-width: 460px; margin: 0 auto; }}
            .btn {{ width: 100%; padding: 14px; border: none; border-radius: 12px; font-size: 1rem; font-weight: bold; cursor: pointer; transition: all 0.2s; }}
            .cam-toggle-group {{ display: flex; gap: 8px; max-width: 460px; margin: 0 auto 10px; }}
            .cam-toggle-btn {{ flex: 1; padding: 10px 12px; border-radius: 10px; background: #1e293b; color: #94a3b8; border: 1.5px solid #334155; font-size: 0.85rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px; transition: all 0.2s; }}
            .cam-toggle-btn.active {{ background: #0369a1; color: #fff; border-color: #38bdf8; box-shadow: 0 0 12px rgba(56, 189, 248, 0.3); }}

            .rot-group {{ display: flex; gap: 6px; max-width: 460px; margin: 0 auto 10px; }}
            .rot-btn {{ flex: 1; padding: 8px; border-radius: 8px; background: #1e293b; color: #94a3b8; border: 1px solid #334155; font-size: 0.78rem; cursor: pointer; }}
            .rot-btn.active {{ background: #0284c7; color: white; border-color: #38bdf8; }}

            .camera-container {{ position: relative; width: 100%; max-width: 460px; margin: 0 auto 12px; border-radius: 16px; overflow: hidden; background: #1e293b; border: 2px solid #334155; aspect-ratio: 4/3; }}
            video {{ width: 100%; height: 100%; object-fit: cover; }}
            video.mirror {{ transform: scaleX(-1); }}
            canvas {{ display: none; }}
            .snap-preview {{ max-width: 100%; border-radius: 8px; margin-top: 10px; display: none; }}

            .stats-badge {{ position: absolute; top: 10px; left: 10px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(6px); border: 1px solid #38bdf8; border-radius: 20px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600; color: #38bdf8; display: flex; align-items: center; gap: 6px; }}
            .live-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #38bdf8; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.4; transform: scale(0.85); }} }}

            .controls {{ display: flex; flex-direction: column; gap: 10px; max-width: 460px; margin: 0 auto; }}
            .snap-row {{ display: flex; gap: 8px; }}
            .btn {{ width: 100%; padding: 13px; border: none; border-radius: 12px; font-size: 0.95rem; font-weight: bold; cursor: pointer; transition: all 0.2s; }}
            .btn-start {{ background: linear-gradient(135deg, #0284c7, #0369a1); color: white; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3); }}
            .btn-snap {{ background: linear-gradient(135deg, #22c55e, #16a34a); color: white; box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3); font-size: 0.88rem; padding: 11px; }}
            .btn-snap-user {{ background: linear-gradient(135deg, #8b5cf6, #7c3aed); color: white; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3); font-size: 0.88rem; padding: 11px; }}
            .btn-stop {{ background: #ef4444; color: white; display: none; }}
            .btn-https {{ background: #ea580c; color: white; margin-top: 8px; }}

            .status-box {{ background: #1e293b; border-radius: 12px; padding: 14px; margin-top: 12px; max-width: 460px; margin-left: auto; margin-right: auto; text-align: left; font-size: 0.85rem; border: 1px solid #334155; }}
            .status-row {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
            .status-label {{ color: #94a3b8; }}
            .status-value {{ font-weight: bold; color: #e2e8f0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⛑️ Helmet Detection Stream</h1>
            <p>Live AI Safety Monitoring</p>
        </div>

        <div id="httpsBanner" class="https-banner" style="display: none;">
            <h3>🔒 Live Video requires HTTPS</h3>
            <p>Mobile browsers (Chrome/Safari) only allow continuous camera streaming over HTTPS. Tap below to switch to HTTPS (if prompted, tap <i>Advanced &rarr; Proceed</i>):</p>
            <button class="btn btn-https" onclick="switchToHttps()">🔒 Switch to HTTPS Stream (Port {https_port})</button>
        </div>

        <div id="permBanner" class="https-banner" style="display: none; background: #450a0a; border-color: #ef4444;">
            <h3 style="color: #fca5a5;">⚠️ Camera Blocked by Browser</h3>
            <p style="margin-bottom: 8px;">The browser did not grant camera access. You have 2 options to proceed:</p>
            <ol style="text-align: left; padding-left: 20px; font-size: 0.8rem; line-height: 1.5; color: #fecaca;">
                <li>Tap the <b>🔒 Lock / Tune icon</b> in the URL bar &rarr; <b>Permissions</b> &rarr; Set <b>Camera: Allow</b>, then refresh.</li>
                <li><b>OR</b> tap <b>"📸 Snap Back"</b> or <b>"🤳 Snap Front"</b> below (opens native camera without permission restrictions).</li>
            </ol>
        </div>

        <!-- Camera Type Selection (Front vs Back) -->
        <div class="cam-toggle-group">
            <button class="cam-toggle-btn active" id="btnCamBack" onclick="selectCameraMode('environment')">
                📷 Back Camera (Main)
            </button>
            <button class="cam-toggle-btn" id="btnCamFront" onclick="selectCameraMode('user')">
                🤳 Front Camera (Selfie)
            </button>
        </div>

        <!-- Orientation Rotation Controls -->
        <div class="rot-group">
            <button class="rot-btn active" id="rot-none" onclick="setOrientation('none')">Normal</button>
            <button class="rot-btn" id="rot-cw" onclick="setOrientation('rotate_90_cw')">🔄 90° CW</button>
            <button class="rot-btn" id="rot-ccw" onclick="setOrientation('rotate_90_ccw')">🔄 90° CCW</button>
            <button class="rot-btn" id="rot-180" onclick="setOrientation('rotate_180')">🔄 180°</button>
            <button class="rot-btn" id="rot-fliph" onclick="setOrientation('flip_h')">🪞 Flip</button>
        </div>

        <div class="camera-container">
            <video id="video" autoplay playsinline muted></video>
            <img id="snapPreview" class="snap-preview" alt="Captured Frame">
            <canvas id="canvas"></canvas>
            <div class="stats-badge" id="liveBadge" style="display: none;">
                <span class="live-dot"></span> <span id="badgeText">STREAMING LIVE</span>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-start" id="startBtn" onclick="startCamera()">📹 Start Live Video Stream</button>
            <button class="btn btn-stop" id="stopBtn" onclick="stopCamera()">⏹️ Stop Stream</button>
            
            <div class="snap-row">
                <button class="btn btn-snap" style="flex:1;" onclick="document.getElementById('nativeCamBack').click()">📸 Snap Back Photo</button>
                <button class="btn btn-snap-user" style="flex:1;" onclick="document.getElementById('nativeCamFront').click()">🤳 Snap Front Selfie</button>
            </div>
            <input type="file" id="nativeCamBack" accept="image/*" capture="environment" style="display: none;" onchange="handleNativeSnap(event)">
            <input type="file" id="nativeCamFront" accept="image/*" capture="user" style="display: none;" onchange="handleNativeSnap(event)">
        </div>

        <div class="status-box">
            <div class="status-row">
                <span class="status-label">Active Camera:</span>
                <span class="status-value" id="activeCamText" style="color: #38bdf8;">Back Camera (Main)</span>
            </div>
            <div class="status-row">
                <span class="status-label">Stream Status:</span>
                <span class="status-value" id="streamStatus" style="color: #94a3b8;">Ready</span>
            </div>
            <div class="status-row">
                <span class="status-label">Persons Detected:</span>
                <span class="status-value" id="personCount">0</span>
            </div>
            <div class="status-row">
                <span class="status-label">Safety Status:</span>
                <span class="status-value" id="safetyStatus">—</span>
            </div>
            <div class="status-row">
                <span class="status-label">Frames Sent:</span>
                <span class="status-value" id="framesSent">0</span>
            </div>
        </div>

        <script>
            let video = document.getElementById('video');
            let canvas = document.getElementById('canvas');
            let snapPreview = document.getElementById('snapPreview');
            let stream = null;
            let streamInterval = null;
            let facingMode = 'environment';
            let frameCount = 0;
            let lastFrameTime = Date.now();

            window.onload = function() {{
                if (window.location.protocol === 'http:' && (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)) {{
                    document.getElementById('httpsBanner').style.display = 'block';
                }}
            }};

            function switchToHttps() {{
                window.location.href = 'https://' + window.location.hostname + ':{https_port}/mobile';
            }}

            async function selectCameraMode(mode) {{
                facingMode = mode;
                document.getElementById('btnCamBack').classList.toggle('active', mode === 'environment');
                document.getElementById('btnCamFront').classList.toggle('active', mode === 'user');
                document.getElementById('activeCamText').textContent = mode === 'user' ? 'Front Camera (Selfie)' : 'Back Camera (Main)';

                if (mode === 'user') {{
                    video.classList.add('mirror');
                }} else {{
                    video.classList.remove('mirror');
                }}

                if (stream) {{
                    await startCamera();
                }}
            }}

            async function setOrientation(mode) {{
                document.querySelectorAll('.rot-btn').forEach(b => b.classList.remove('active'));
                const btn = document.getElementById('rot-' + (mode === 'rotate_90_cw' ? 'cw' : mode === 'rotate_90_ccw' ? 'ccw' : mode === 'rotate_180' ? '180' : mode === 'flip_h' ? 'fliph' : 'none'));
                if (btn) btn.classList.add('active');

                try {{
                    await fetch('/camera/transform?transform=' + encodeURIComponent(mode), {{ method: 'POST' }});
                }} catch(e) {{}}
            }}

            async function startCamera() {{
                document.getElementById('permBanner').style.display = 'none';
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
                    document.getElementById('httpsBanner').style.display = 'block';
                    switchToHttps();
                    return;
                }}
                try {{
                    if (stream) {{
                        stream.getTracks().forEach(t => t.stop());
                        stream = null;
                    }}
                    video.srcObject = null;

                    try {{
                        stream = await navigator.mediaDevices.getUserMedia({{
                            video: {{
                                facingMode: {{ exact: facingMode }},
                                width: {{ ideal: 640 }},
                                height: {{ ideal: 480 }}
                            }},
                            audio: false
                        }});
                    }} catch(eExact) {{
                        try {{
                            stream = await navigator.mediaDevices.getUserMedia({{
                                video: {{
                                    facingMode: {{ ideal: facingMode }},
                                    width: {{ ideal: 640 }},
                                    height: {{ ideal: 480 }}
                                }},
                                audio: false
                            }});
                        }} catch(eIdeal) {{
                            stream = await navigator.mediaDevices.getUserMedia({{ video: true, audio: false }});
                        }}
                    }}

                    video.style.display = 'block';
                    snapPreview.style.display = 'none';
                    video.srcObject = stream;
                    await video.play();

                    document.getElementById('startBtn').style.display = 'none';
                    document.getElementById('stopBtn').style.display = 'block';
                    document.getElementById('liveBadge').style.display = 'flex';
                    document.getElementById('streamStatus').textContent = 'Streaming Live';
                    document.getElementById('streamStatus').style.color = '#38bdf8';

                    if (streamInterval) clearInterval(streamInterval);
                    streamInterval = setInterval(sendFrame, 40);
                }} catch(e) {{
                    document.getElementById('permBanner').style.display = 'block';
                    document.getElementById('streamStatus').textContent = 'Permission Denied / Insecure Context';
                    document.getElementById('streamStatus').style.color = '#ef4444';
                }}
            }}

            function stopCamera() {{
                if (streamInterval) {{
                    clearInterval(streamInterval);
                    streamInterval = null;
                }}
                if (stream) {{
                    stream.getTracks().forEach(t => t.stop());
                    stream = null;
                }}
                video.srcObject = null;
                document.getElementById('startBtn').style.display = 'block';
                document.getElementById('stopBtn').style.display = 'none';
                document.getElementById('liveBadge').style.display = 'none';
                document.getElementById('streamStatus').textContent = 'Stopped';
                document.getElementById('streamStatus').style.color = '#94a3b8';
            }}

            async function switchCamera() {{
                facingMode = (facingMode === 'environment') ? 'user' : 'environment';
                if (stream) await startCamera();
            }}

            let isSending = false;
            function sendFrame() {{
                if (!video.videoWidth || isSending || !stream) return;
                canvas.width = Math.min(video.videoWidth, 640);
                canvas.height = Math.min(video.videoHeight, 480);
                let ctx = canvas.getContext('2d');

                // If user camera, draw mirrored on canvas if needed
                if (facingMode === 'user') {{
                    ctx.save();
                    ctx.translate(canvas.width, 0);
                    ctx.scale(-1, 1);
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    ctx.restore();
                }} else {{
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                }}

                isSending = true;

                canvas.toBlob(blob => {{
                    if (!blob) {{ isSending = false; return; }}
                    let fd = new FormData();
                    fd.append('file', blob, 'frame.jpg');
                    fetch('/ingest/frame', {{ method: 'POST', body: fd }})
                        .then(r => r.json())
                        .then(data => {{
                            isSending = false;
                            frameCount++;
                            document.getElementById('framesSent').textContent = frameCount;
                            document.getElementById('personCount').textContent = data.num_persons || 0;
                            let viol = data.num_violations || 0;
                            let safetyEl = document.getElementById('safetyStatus');
                            if (viol > 0) {{
                                safetyEl.textContent = 'VIOLATION (' + viol + ' No Helmet)';
                                safetyEl.style.color = '#ef4444';
                            }} else if ((data.num_persons || 0) > 0) {{
                                safetyEl.textContent = 'SAFE (' + (data.num_helmets || 0) + ' Helmet)';
                                safetyEl.style.color = '#4ade80';
                            }} else {{
                                safetyEl.textContent = 'No Persons';
                                safetyEl.style.color = '#94a3b8';
                            }}
                        }})
                        .catch(err => {{ isSending = false; }});
                }}, 'image/jpeg', 0.70);
            }}

            function handleNativeSnap(event) {{
                const file = event.target.files[0];
                if (!file) return;
                const fd = new FormData();
                fd.append('file', file);
                document.getElementById('streamStatus').textContent = 'Uploading snapshot...';
                document.getElementById('streamStatus').style.color = '#38bdf8';

                fetch('/ingest/frame', {{ method: 'POST', body: fd }})
                    .then(r => r.json())
                    .then(data => {{
                        frameCount++;
                        document.getElementById('framesSent').textContent = frameCount;
                        document.getElementById('streamStatus').textContent = 'Snapshot Processed ✅';
                        document.getElementById('streamStatus').style.color = '#4ade80';
                        document.getElementById('personCount').textContent = data.num_persons || 0;

                        const reader = new FileReader();
                        reader.onload = e => {{
                            snapPreview.src = e.target.result;
                            snapPreview.style.display = 'block';
                            video.style.display = 'none';
                        }};
                        reader.readAsDataURL(file);
                    }})
                    .catch(err => {{
                        document.getElementById('streamStatus').textContent = 'Upload failed ❌';
                        document.getElementById('streamStatus').style.color = '#ef4444';
                    }});
            }}
        </script>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def root():
    local_ip = _get_local_ip()
    port = int(os.getenv("PORT", "8002"))
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Helmet Detection — Live Monitor</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: #090d16; color: #f8fafc; display: flex; flex-direction: column; align-items: center; min-height: 100vh; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}

            .topbar {{
                width: 100%;
                background: #0f172a;
                border-bottom: 1px solid #1e293b;
                padding: 14px 28px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .topbar-title {{ font-size: 1.2rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
            .status-indicator {{ display: flex; align-items: center; gap: 8px; font-size: 0.9rem; color: #94a3b8; }}
            .status-dot {{ width: 10px; height: 10px; border-radius: 50%; background: #ef4444; }}
            .status-dot.live {{ background: #22c55e; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}

            .main-content {{ width: 100%; max-width: 1080px; padding: 20px; display: flex; flex-direction: column; gap: 18px; }}

            /* Violation Banner */
            .alert-banner {{
                display: none;
                background: linear-gradient(90deg, #991b1b, #dc2626);
                color: #fff;
                padding: 14px 20px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 1.05rem;
                text-align: center;
                animation: flash 1s infinite alternate;
                box-shadow: 0 4px 20px rgba(220, 38, 38, 0.4);
            }}
            @keyframes flash {{ from {{ opacity: 0.85; }} to {{ opacity: 1; }} }}

            /* Video Stream */
            .video-container {{
                position: relative;
                width: 100%;
                background: #020617;
                border-radius: 16px;
                overflow: hidden;
                border: 2px solid #1e293b;
                aspect-ratio: 16/9;
                max-height: 580px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            #liveStream {{
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
            }}
            .waiting-overlay {{
                position: absolute;
                inset: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background: #020617;
                color: #64748b;
                gap: 14px;
            }}
            .waiting-overlay.hidden {{ display: none; }}
            .spinner {{ width: 44px; height: 44px; border: 4px solid #1e293b; border-top-color: #38bdf8; border-radius: 50%; animation: spin 1s linear infinite; }}
            @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

            /* KPI Cards */
            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 14px;
            }}
            .kpi-card {{
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 14px;
                padding: 16px 20px;
                text-align: center;
            }}
            .kpi-value {{ font-size: 2.5rem; font-weight: 800; line-height: 1.1; }}
            .kpi-label {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-top: 6px; }}
            .val-total {{ color: #38bdf8; }}
            .val-safe {{ color: #4ade80; }}
            .val-violation {{ color: #f87171; }}

            /* Mobile Connection Info */
            .connection-bar {{
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 14px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.9rem;
            }}
            .connection-bar a {{ color: #38bdf8; font-weight: bold; text-decoration: none; }}
            .connection-bar a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="topbar">
            <div class="topbar-title">⛑️ Helmet Detection — Live AI Monitor</div>
            <div class="status-indicator">
                <span class="status-dot" id="statusDot"></span>
                <span id="statusText">Waiting for camera...</span>
            </div>
        </div>

        <div class="main-content">
            <div class="alert-banner" id="alertBanner">
                ⚠️ VIOLATION DETECTED: PERSON WITHOUT SAFETY HELMET!
            </div>

            <div class="video-container">
                <img id="liveStream" src="/stream/detect" alt="live feed">
                <div class="waiting-overlay" id="waitingOverlay">
                    <div class="spinner"></div>
                    <span style="font-size: 1.1rem; font-weight: 500;">Waiting for mobile camera stream...</span>
                    <small>Open <b>/mobile</b> on smartphone to stream live video</small>
                </div>
            </div>

            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-value val-total" id="kpiTotal">0</div>
                    <div class="kpi-label">Total Persons</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value val-safe" id="kpiSafe">0</div>
                    <div class="kpi-label">Helmets Worn (Safe)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value val-violation" id="kpiViolations">0</div>
                    <div class="kpi-label">Violations (No Helmet)</div>
                </div>
            </div>

            <div class="connection-bar">
                <div>
                    <span>📱 <b>Connect Mobile Camera (HTTPS Public Link):</b></span><br>
                    <a id="tunnelLink" href="https://older-reflected-allowed-hydrogen.trycloudflare.com/mobile" target="_blank" style="font-size: 1.05rem; color: #38bdf8;">
                        https://older-reflected-allowed-hydrogen.trycloudflare.com/mobile
                    </a>
                </div>
                <div style="text-align: right; color: #64748b; font-size: 0.8rem;">
                    Local Wi-Fi: <a href="http://{local_ip}:{port}/mobile" target="_blank" style="color: #94a3b8;">http://{local_ip}:{port}/mobile</a>
                </div>
            </div>
        </div>

        <script>
            let alertBanner = document.getElementById('alertBanner');
            let waitingOverlay = document.getElementById('waitingOverlay');
            let statusDot = document.getElementById('statusDot');
            let statusText = document.getElementById('statusText');

            async function pollStatus() {{
                try {{
                    let res = await fetch('/stream/status');
                    let data = await res.json();

                    if (data.ingest_active) {{
                        waitingOverlay.classList.add('hidden');
                        statusDot.classList.add('live');
                        statusText.textContent = 'Live (' + data.ingest_frames + ' frames)';
                    }} else {{
                        waitingOverlay.classList.remove('hidden');
                        statusDot.classList.remove('live');
                        statusText.textContent = 'Waiting for camera...';
                    }}

                    let d = data.latest_detection || {{}};
                    let total = d.num_persons || 0;
                    let safe = d.num_helmets || 0;
                    let violations = d.num_violations || 0;

                    document.getElementById('kpiTotal').textContent = total;
                    document.getElementById('kpiSafe').textContent = safe;
                    document.getElementById('kpiViolations').textContent = violations;

                    if (violations > 0 && data.ingest_active) {{
                        alertBanner.style.display = 'block';
                        alertBanner.textContent = '⚠️ VIOLATION DETECTED: ' + violations + ' PERSON(S) WITHOUT SAFETY HELMET!';
                    }} else {{
                        alertBanner.style.display = 'none';
                    }}
                }} catch (e) {{
                    console.error('Status poll error', e);
                }}
            }}

            setInterval(pollStatus, 400);
            pollStatus();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    http_port = int(os.getenv("PORT", "8002"))
    print(f"Starting Helmet Detection HTTP server on http://0.0.0.0:{http_port}")
    uvicorn.run(app, host="0.0.0.0", port=http_port, log_level="info")

