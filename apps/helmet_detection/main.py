"""FastAPI app: Helmet Detection (App 2).

Live video from USB/Wired Mobile Camera, IP Webcam / DroidCam via ADB,
or Mobile Browser push. Performs real-time AI safety helmet compliance detection.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from camera_manager import CameraManager, get_orientation_transform, list_system_cameras, set_orientation_transform
from detector import FrameResult, HelmetDetector, PersonStatus, draw_helmet_detections
from streamer import FrameBuffer, apply_transform, mjpeg_from_buffer

load_dotenv(find_dotenv(usecwd=True))

app = FastAPI(
    title="Helmet Detection",
    description="Detect persons with/without safety helmets from live wired/wireless camera feeds",
    version="1.3.0",
)

_detector: Optional[HelmetDetector] = None
_camera_manager: Optional[CameraManager] = None
_violations_log: deque = deque(maxlen=200)

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


def _get_camera_manager() -> CameraManager:
    global _camera_manager
    if _camera_manager is None:
        _camera_manager = CameraManager()
    return _camera_manager


def _get_local_ip() -> str:
    lan = os.getenv("LAN_IP", "").strip()
    if lan:
        return lan
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
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
    """Continuous background worker running AI inference on fresh video frames."""
    global _cached_result, _last_detection_info
    last_seen_ts = 0.0

    while _worker_running:
        try:
            cam_mgr = _get_camera_manager()
            buf = cam_mgr.camera.buffer

            if not buf.is_active:
                time.sleep(0.04)
                continue

            latest = buf.get_latest()
            if latest is None:
                time.sleep(0.04)
                continue

            ts, frame = latest
            if ts != last_seen_ts:
                last_seen_ts = ts
                conf = float(os.getenv("CONF_THRESHOLD", "0.35"))
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
        except Exception:
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

            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-keyout",
                    key_file,
                    "-out",
                    cert_file,
                    "-days",
                    "365",
                    "-nodes",
                    "-subj",
                    "/CN=helmet-detection",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
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
    cam_mgr = _get_camera_manager()
    status = cam_mgr.get_status()
    return {
        "status": "healthy",
        "backend": detector.backend,
        "camera": status["health"],
        "orientation": status["orientation"],
    }


@app.get("/model/info")
async def model_info():
    return _get_detector().get_model_info()


@app.post("/detect")
async def detect(file: UploadFile = File(...), confidence: Optional[float] = Query(default=None)):
    image = _read_image(await file.read())
    conf = confidence if confidence is not None else float(os.getenv("CONF_THRESHOLD", "0.35"))
    result = _get_detector().detect(image, confidence=conf)
    return result.to_dict()


@app.post("/detect/visualize")
async def detect_visualize(file: UploadFile = File(...), confidence: Optional[float] = Query(default=None)):
    image = _read_image(await file.read())
    conf = confidence if confidence is not None else float(os.getenv("CONF_THRESHOLD", "0.35"))
    result = _get_detector().detect(image, confidence=conf)
    vis = draw_helmet_detections(image, result)
    _, buffer = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/ingest/frame")
async def ingest_frame(file: UploadFile = File(...)):
    """Push one JPEG frame from the mobile browser."""
    image = _read_image(await file.read())
    _get_camera_manager().camera.ingest_push_frame(image)

    with _detection_lock:
        info = dict(_last_detection_info)

    return {
        "status": "accepted",
        "ingest_frames": _get_camera_manager().camera.health.frame_count,
        "num_persons": info.get("num_persons", 0),
        "num_helmets": info.get("num_helmets", 0),
        "num_violations": info.get("num_violations", 0),
        "size": [image.shape[1], image.shape[0]],
    }


@app.get("/violations")
async def violations(limit: int = Query(default=50, ge=1, le=200)):
    """Recent no-helmet events log."""
    items = list(_violations_log)[-limit:]
    return {"count": len(items), "violations": items}


@app.get("/usb/cameras")
async def list_usb_cameras_endpoint():
    """List available Linux /dev/video* devices without opening or locking them."""
    cameras = list_system_cameras()
    return {"cameras": cameras}


@app.post("/usb/start")
async def usb_start(
    device_index: int = Query(default=0, description="USB device index (0=/dev/video0, 2=/dev/video2)"),
    fps: int = Query(default=25),
    width: int = Query(default=1280),
    height: int = Query(default=720),
):
    """Start capturing from a wired USB / Mobile webcam."""
    cam_mgr = _get_camera_manager()
    ok, msg = cam_mgr.camera.connect(
        source_type="usb",
        uri=f"/dev/video{device_index}",
        fps_target=fps,
        width=width,
        height=height,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=f"Failed to start USB camera: {msg}")
    return {"status": "started", "device": f"/dev/video{device_index}", "message": msg}


@app.post("/usb/stop")
async def usb_stop():
    """Stop USB webcam capture."""
    cam_mgr = _get_camera_manager()
    cam_mgr.camera.disconnect()
    return {"status": "stopped"}


@app.post("/camera/connect")
async def camera_connect(
    source_uri: str = Query(..., description="e.g. http://127.0.0.1:8080/video or http://127.0.0.1:4747/video or rtsp://..."),
    source_type: str = Query(default="http_mjpeg", description="http_mjpeg | rtsp | usb"),
    fps: int = Query(default=20),
):
    """Connect to a wired ADB stream, DroidCam, IP Webcam, or RTSP feed."""
    cam_mgr = _get_camera_manager()
    ok, msg = cam_mgr.camera.connect(
        source_type=source_type,
        uri=source_uri.strip(),
        fps_target=fps,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=f"Failed to connect to camera source: {msg}")
    return {"status": "connected", "source": source_uri, "source_type": source_type}


@app.post("/camera/disconnect")
async def camera_disconnect():
    """Disconnect active camera capture."""
    cam_mgr = _get_camera_manager()
    cam_mgr.camera.disconnect()
    return {"status": "disconnected"}


@app.post("/camera/transform")
async def set_camera_transform_endpoint(transform: str = Query(default="none")):
    """Live orientation adjustment: none | flip_h | flip_v | rotate_90_cw | rotate_90_ccw | rotate_180."""
    valid = ("none", "flip_h", "flip_v", "rotate_90_cw", "rotate_90_ccw", "rotate_180", "rotate_90", "rotate_270")
    t_clean = transform.lower().strip()
    if t_clean not in valid:
        raise HTTPException(status_code=422, detail=f"Invalid transform '{transform}'. Must be one of: {valid}")
    if t_clean in ("rotate_90", "90"):
        t_clean = "rotate_90_cw"
    elif t_clean in ("rotate_270", "270"):
        t_clean = "rotate_90_ccw"

    set_orientation_transform(t_clean)
    return {"status": "ok", "transform": get_orientation_transform()}


@app.get("/stream")
async def stream():
    """Raw MJPEG stream from the active camera buffer."""
    cam_mgr = _get_camera_manager()
    gen = mjpeg_from_buffer(cam_mgr.camera.buffer, quality=55, fps_limit=30.0)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stream/detect")
async def stream_detect():
    """Live annotated MJPEG with AI safety bounding boxes (Green=Safe Helmet, Red=No Helmet Violation)."""
    cam_mgr = _get_camera_manager()

    def annotate(frame: np.ndarray) -> np.ndarray:
        with _detection_lock:
            cached = _cached_result
        if cached is not None:
            return draw_helmet_detections(frame, cached)
        return frame

    gen = mjpeg_from_buffer(cam_mgr.camera.buffer, quality=55, transform=annotate, fps_limit=30.0)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stream/status")
async def stream_status():
    """Real-time system telemetry and camera health status."""
    local_ip = _get_local_ip()
    port = int(os.getenv("PORT", "8002"))
    https_port = int(os.getenv("HTTPS_PORT", "8444"))
    cam_mgr = _get_camera_manager()
    status = cam_mgr.get_status()

    with _detection_lock:
        info = dict(_last_detection_info)

    is_active = status["health"]["connected"] or cam_mgr.camera.buffer.is_active

    return {
        "ingest_active": is_active,
        "ingest_frames": status["health"]["frame_count"],
        "camera_health": status["health"],
        "latest_detection": info,
        "transform": status["orientation"],
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
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b1120; color: #f8fafc; padding: 15px; text-align: center; }}
            .header {{ margin-bottom: 12px; }}
            h1 {{ font-size: 1.4rem; color: #38bdf8; margin-bottom: 4px; font-weight: 800; }}
            p {{ font-size: 0.85rem; color: #94a3b8; }}
            
            .https-banner {{ background: #7c2d12; border: 1px solid #ea580c; border-radius: 12px; padding: 12px; margin-bottom: 15px; text-align: left; font-size: 0.85rem; }}
            .https-banner h3 {{ color: #fdba74; font-size: 0.95rem; margin-bottom: 4px; }}
            
            .camera-container {{ position: relative; width: 100%; max-width: 480px; margin: 0 auto 12px; border-radius: 16px; overflow: hidden; background: #1e293b; border: 2px solid #334155; aspect-ratio: 4/3; }}
            video {{ width: 100%; height: 100%; object-fit: cover; }}
            video.mirror {{ transform: scaleX(-1); }}
            canvas {{ display: none; }}
            .snap-preview {{ max-width: 100%; border-radius: 8px; margin-top: 10px; display: none; }}
            
            .stats-badge {{ position: absolute; top: 10px; left: 10px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(6px); border: 1px solid #38bdf8; border-radius: 20px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600; color: #38bdf8; display: flex; align-items: center; gap: 6px; }}
            .live-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #38bdf8; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.4; transform: scale(0.85); }} }}
            
            .controls {{ display: flex; flex-direction: column; gap: 10px; max-width: 480px; margin: 0 auto; }}
            .btn {{ width: 100%; padding: 14px; border: none; border-radius: 12px; font-size: 1rem; font-weight: bold; cursor: pointer; transition: all 0.2s; }}
            .cam-toggle-group {{ display: flex; gap: 8px; max-width: 480px; margin: 0 auto 10px; }}
            .cam-toggle-btn {{ flex: 1; padding: 10px 12px; border-radius: 10px; background: #1e293b; color: #94a3b8; border: 1.5px solid #334155; font-size: 0.85rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px; transition: all 0.2s; }}
            .cam-toggle-btn.active {{ background: #0369a1; color: #fff; border-color: #38bdf8; box-shadow: 0 0 12px rgba(56, 189, 248, 0.3); }}

            .rot-group {{ display: flex; gap: 6px; max-width: 480px; margin: 0 auto 10px; }}
            .rot-btn {{ flex: 1; padding: 8px; border-radius: 8px; background: #1e293b; color: #94a3b8; border: 1px solid #334155; font-size: 0.78rem; cursor: pointer; }}
            .rot-btn.active {{ background: #0284c7; color: white; border-color: #38bdf8; }}

            .snap-row {{ display: flex; gap: 8px; }}
            .btn-start {{ background: linear-gradient(135deg, #0284c7, #0369a1); color: white; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3); }}
            .btn-snap {{ background: linear-gradient(135deg, #22c55e, #16a34a); color: white; box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3); font-size: 0.88rem; padding: 11px; }}
            .btn-snap-user {{ background: linear-gradient(135deg, #8b5cf6, #7c3aed); color: white; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3); font-size: 0.88rem; padding: 11px; }}
            .btn-stop {{ background: #ef4444; color: white; display: none; }}
            .btn-https {{ background: #ea580c; color: white; margin-top: 8px; }}

            .status-box {{ background: #1e293b; border-radius: 12px; padding: 14px; margin-top: 12px; max-width: 480px; margin-left: auto; margin-right: auto; text-align: left; font-size: 0.85rem; border: 1px solid #334155; }}
            .status-row {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
            .status-label {{ color: #94a3b8; }}
            .status-value {{ font-weight: bold; color: #e2e8f0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⛑️ Helmet Detection Stream</h1>
            <p>Live Mobile Video Feed for AI Safety</p>
        </div>

        <div id="httpsBanner" class="https-banner" style="display: none;">
            <h3>🔒 Live Video requires HTTPS</h3>
            <p>Mobile browsers (Chrome/Safari) only allow continuous camera streaming over HTTPS. Tap below to switch to HTTPS:</p>
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

        <!-- Camera Type Selection -->
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
                <span class="status-label">Safety Compliance:</span>
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
                    document.getElementById('streamStatus').textContent = 'Permission Denied';
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

            let isSending = false;
            function sendFrame() {{
                if (!video.videoWidth || isSending || !stream) return;
                canvas.width = Math.min(video.videoWidth, 640);
                canvas.height = Math.min(video.videoHeight, 480);
                let ctx = canvas.getContext('2d');

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
                }}, 'image/jpeg', 0.75);
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
    https_port = int(os.getenv("HTTPS_PORT", "8444"))
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Helmet Detection — Live AI Monitor & Wired Camera</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: #070b14; color: #f8fafc; display: flex; flex-direction: column; align-items: center; min-height: 100vh; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}

            .topbar {{
                width: 100%;
                background: #0f172a;
                border-bottom: 1px solid #1e293b;
                padding: 14px 28px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            .topbar-title {{ font-size: 1.25rem; font-weight: 800; color: #38bdf8; display: flex; align-items: center; gap: 10px; letter-spacing: 0.5px; }}
            .status-indicator {{ display: flex; align-items: center; gap: 10px; font-size: 0.9rem; color: #94a3b8; background: #1e293b; padding: 6px 14px; border-radius: 20px; border: 1px solid #334155; }}
            .status-dot {{ width: 10px; height: 10px; border-radius: 50%; background: #ef4444; }}
            .status-dot.live {{ background: #22c55e; box-shadow: 0 0 10px #22c55e; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}

            .main-content {{ width: 100%; max-width: 1120px; padding: 20px; display: flex; flex-direction: column; gap: 18px; }}

            /* Violation Banner */
            .alert-banner {{
                display: none;
                background: linear-gradient(90deg, #991b1b, #dc2626);
                color: #fff;
                padding: 14px 20px;
                border-radius: 12px;
                font-weight: 800;
                font-size: 1.05rem;
                text-align: center;
                animation: flash 1s infinite alternate;
                box-shadow: 0 4px 20px rgba(220, 38, 38, 0.4);
                border: 1px solid #f87171;
            }}
            @keyframes flash {{ from {{ opacity: 0.85; }} to {{ opacity: 1; }} }}

            /* Video Stream Section */
            .video-container {{
                position: relative;
                width: 100%;
                background: #020617;
                border-radius: 16px;
                overflow: hidden;
                border: 2px solid #1e293b;
                aspect-ratio: 16/9;
                max-height: 560px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 8px 30px rgba(0,0,0,0.5);
            }}
            #liveStream {{
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
            }}
            .video-overlay-badge {{
                position: absolute;
                top: 14px;
                left: 14px;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(8px);
                border: 1px solid #38bdf8;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 0.82rem;
                font-weight: 700;
                color: #38bdf8;
                display: flex;
                align-items: center;
                gap: 8px;
                z-index: 10;
            }}
            .waiting-overlay {{
                position: absolute;
                inset: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background: rgba(2, 6, 23, 0.95);
                color: #94a3b8;
                gap: 14px;
                z-index: 5;
            }}
            .waiting-overlay.hidden {{ display: none; }}
            .spinner {{ width: 44px; height: 44px; border: 4px solid #1e293b; border-top-color: #38bdf8; border-radius: 50%; animation: spin 1s linear infinite; }}
            @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

            /* Orientation Controls Toolbar */
            .orientation-bar {{
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 10px 16px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .rot-btn-group {{ display: flex; gap: 6px; flex-wrap: wrap; }}
            .btn-rot {{
                background: #1e293b;
                color: #cbd5e1;
                border: 1px solid #334155;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 0.82rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.15s;
            }}
            .btn-rot:hover {{ background: #334155; color: #fff; }}
            .btn-rot.active {{ background: #0284c7; color: white; border-color: #38bdf8; box-shadow: 0 0 10px rgba(56, 189, 248, 0.4); }}

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
                padding: 18px 20px;
                text-align: center;
                box-shadow: 0 4px 14px rgba(0,0,0,0.2);
            }}
            .kpi-value {{ font-size: 2.6rem; font-weight: 800; line-height: 1.1; }}
            .kpi-label {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-top: 6px; font-weight: 600; }}
            .val-total {{ color: #38bdf8; }}
            .val-safe {{ color: #4ade80; }}
            .val-violation {{ color: #f87171; }}

            /* Connection Control Center (Tabs) */
            .control-center {{
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }}
            .tab-nav {{
                display: flex;
                gap: 10px;
                margin-bottom: 16px;
                border-bottom: 1px solid #1e293b;
                padding-bottom: 12px;
            }}
            .tab-btn {{
                flex: 1;
                padding: 12px 16px;
                border-radius: 10px;
                border: none;
                cursor: pointer;
                font-weight: 700;
                font-size: 0.92rem;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                transition: all 0.2s;
                background: #1e293b;
                color: #94a3b8;
            }}
            .tab-btn.active {{
                background: #0284c7;
                color: #ffffff;
                box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
            }}

            .tab-panel {{ display: none; }}
            .tab-panel.active {{ display: block; }}

            .form-row {{
                display: flex;
                gap: 10px;
                align-items: center;
                margin-bottom: 12px;
                flex-wrap: wrap;
            }}
            .input-text, .select-custom {{
                flex: 1;
                background: #020617;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 0.9rem;
                outline: none;
            }}
            .input-text:focus, .select-custom:focus {{ border-color: #38bdf8; }}
            
            .btn-action {{
                padding: 10px 18px;
                border-radius: 8px;
                border: none;
                font-weight: 700;
                font-size: 0.9rem;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 6px;
                transition: all 0.2s;
            }}
            .btn-primary {{ background: #16a34a; color: white; }}
            .btn-primary:hover {{ background: #15803d; }}
            .btn-danger {{ background: #dc2626; color: white; }}
            .btn-danger:hover {{ background: #b91c1c; }}
            .btn-info {{ background: #0284c7; color: white; }}
            .btn-info:hover {{ background: #0369a1; }}

            .preset-chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
            .preset-chip {{
                background: #1e293b;
                border: 1px solid #334155;
                color: #38bdf8;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 0.8rem;
                cursor: pointer;
                font-weight: 600;
            }}
            .preset-chip:hover {{ background: #334155; }}

            .guide-accordion {{
                margin-top: 14px;
                background: #020617;
                border: 1px solid #1e293b;
                border-radius: 10px;
                padding: 12px 16px;
                font-size: 0.85rem;
                color: #cbd5e1;
                line-height: 1.6;
            }}
            .guide-title {{ font-weight: 700; color: #38bdf8; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }}

            .status-log {{
                background: #020617;
                border-radius: 8px;
                padding: 10px 14px;
                font-family: monospace;
                font-size: 0.82rem;
                margin-top: 10px;
                min-height: 38px;
                display: flex;
                align-items: center;
            }}
        </style>
    </head>
    <body>
        <div class="topbar">
            <div class="topbar-title">⛑️ Helmet Detection — Live AI Safety Monitor</div>
            <div class="status-indicator">
                <span class="status-dot" id="statusDot"></span>
                <span id="statusText">Waiting for Camera...</span>
            </div>
        </div>

        <div class="main-content">
            <!-- Alert Banner -->
            <div class="alert-banner" id="alertBanner">
                ⚠️ VIOLATION DETECTED: PERSON WITHOUT SAFETY HELMET!
            </div>

            <!-- Video Monitor -->
            <div class="video-container">
                <div class="video-overlay-badge" id="hudBadge">
                    <span>📷 <span id="hudSource">No Source</span></span>
                    <span style="opacity: 0.4;">|</span>
                    <span id="hudFps">0.0 FPS</span>
                    <span style="opacity: 0.4;">|</span>
                    <span id="hudRes">0x0</span>
                </div>
                <img id="liveStream" src="/stream/detect" alt="Live Annotated Feed">
                <div class="waiting-overlay" id="waitingOverlay">
                    <div class="spinner"></div>
                    <span style="font-size: 1.15rem; font-weight: 600; color: #f1f5f9;">Waiting for Live Camera Stream...</span>
                    <small style="color: #94a3b8;">Plug in USB Wire below OR connect IP/Mobile stream to begin detection</small>
                </div>
            </div>

            <!-- Orientation Correction Toolbar -->
            <div class="orientation-bar">
                <span style="font-size: 0.85rem; font-weight: 700; color: #94a3b8;">🔄 Camera Orientation / Rotation:</span>
                <div class="rot-btn-group">
                    <button class="btn-rot active" id="rot-none" onclick="setOrientation('none')">Normal (0°)</button>
                    <button class="btn-rot" id="rot-cw" onclick="setOrientation('rotate_90_cw')">🔄 90° CW</button>
                    <button class="btn-rot" id="rot-ccw" onclick="setOrientation('rotate_90_ccw')">🔄 90° CCW</button>
                    <button class="btn-rot" id="rot-180" onclick="setOrientation('rotate_180')">🔄 180°</button>
                    <button class="btn-rot" id="rot-fliph" onclick="setOrientation('flip_h')">🪞 Flip Horiz</button>
                </div>
            </div>

            <!-- KPI Counters -->
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

            <!-- Camera Connection Hub (Wired USB / IP ADB / Mobile) -->
            <div class="control-center">
                <div class="tab-nav">
                    <button class="tab-btn active" id="tabBtnWired" onclick="switchTab('wired')">
                        🔌 Wired USB / Mobile Camera
                    </button>
                    <button class="tab-btn" id="tabBtnNet" onclick="switchTab('net')">
                        🔗 Wired ADB / IP Webcam / DroidCam
                    </button>
                    <button class="tab-btn" id="tabBtnMobile" onclick="switchTab('mobile')">
                        📱 Wireless Mobile (Browser)
                    </button>
                </div>

                <!-- Tab 1: Wired USB / Mobile V4L2 -->
                <div class="tab-panel active" id="panelWired">
                    <div style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 12px;">
                        Connect your mobile phone with a USB cable (Android 14+ Webcam mode, DroidCam, or USB camera) directly to this computer.
                    </div>

                    <div class="form-row">
                        <button class="btn-action btn-info" onclick="detectUsbCams()">🔍 Detect USB Devices</button>
                        <select class="select-custom" id="usbCameraSelect" style="max-width: 480px;">
                            <option value="0">/dev/video0 (Default)</option>
                            <option value="1">/dev/video1</option>
                            <option value="2">/dev/video2</option>
                            <option value="4">/dev/video4</option>
                        </select>
                        <button class="btn-action btn-primary" onclick="startUsbCam()">▶ Start Live Camera</button>
                        <button class="btn-action btn-danger" onclick="stopCamera()">■ Disconnect</button>
                    </div>

                    <div class="status-log" id="usbStatusLog" style="color: #94a3b8;">
                        Click "Detect USB Devices" to scan connected mobile / USB cameras.
                    </div>

                    <div class="guide-accordion">
                        <div class="guide-title">📖 Mobile Phone ko USB Wire sy Connect krnay ka Tareeqa (3 Easy Options):</div>
                        <p><b>Option 1 (Android 14+ Built-in):</b> Mobile ko USB cable sy laptop mein lagayein &rarr; Phone notification mein "USB Preferences" par tap kr k <b>"Webcam"</b> select karein &rarr; Upar <b>"Detect USB Devices"</b> dabayein aur <b>Start</b> karein.</p>
                        <p><b>Option 2 (DroidCam USB):</b> Mobile mein DroidCam install karein &rarr; USB Debugging on karein &rarr; USB cable lagayein &rarr; Upar select karein.</p>
                        <p><b>Option 3 (IP Webcam via USB Tethering):</b> Phone mein "USB Tethering" on karein &rarr; IP Webcam app chala kar "Tab 2 (Wired ADB/IP)" se connect karein.</p>
                    </div>
                </div>

                <!-- Tab 2: IP Webcam / ADB Port Forwarding -->
                <div class="tab-panel" id="panelNet">
                    <div style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 12px;">
                        Stream directly from IP Webcam or DroidCam over USB cable using ADB port forwarding or USB Tethering.
                    </div>

                    <div class="form-row">
                        <input type="text" class="input-text" id="streamUri" placeholder="e.g. http://127.0.0.1:8080/video or http://127.0.0.1:4747/video" value="http://127.0.0.1:8080/video">
                        <button class="btn-action btn-primary" onclick="connectIpCam()">🔗 Connect Stream</button>
                        <button class="btn-action btn-danger" onclick="stopCamera()">■ Disconnect</button>
                    </div>

                    <div class="preset-chips">
                        <span style="font-size: 0.8rem; color: #64748b; align-self: center;">Quick Presets:</span>
                        <div class="preset-chip" onclick="setPreset('http://127.0.0.1:8080/video')">⚡ IP Webcam ADB (8080)</div>
                        <div class="preset-chip" onclick="setPreset('http://127.0.0.1:4747/video')">⚡ DroidCam USB (4747)</div>
                        <div class="preset-chip" onclick="setPreset('http://192.168.42.129:8080/video')">⚡ USB Tethering IP</div>
                    </div>

                    <div class="status-log" id="netStatusLog" style="color: #94a3b8;">
                        Enter stream URL and click Connect.
                    </div>
                </div>

                <!-- Tab 3: Wireless Mobile Browser -->
                <div class="tab-panel" id="panelMobile">
                    <div style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 12px;">
                        Open the mobile camera page on your phone browser to stream directly over HTTPS:
                    </div>

                    <div style="background: #020617; border-radius: 10px; padding: 14px; border: 1px solid #1e293b;">
                        <div style="margin-bottom: 8px;">
                            <span style="color: #94a3b8;">🔒 Public HTTPS Cloudflare Tunnel:</span><br>
                            <a id="tunnelLink" href="/mobile" target="_blank" style="color: #38bdf8; font-size: 1.05rem; font-weight: bold; text-decoration: none;">
                                /mobile
                            </a>
                        </div>
                        <div style="font-size: 0.82rem; color: #64748b;">
                            Local Wi-Fi: <a href="http://{local_ip}:{port}/mobile" target="_blank" style="color: #94a3b8;">http://{local_ip}:{port}/mobile</a> | HTTPS: <a href="https://{local_ip}:{https_port}/mobile" target="_blank" style="color: #94a3b8;">https://{local_ip}:{https_port}/mobile</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let alertBanner = document.getElementById('alertBanner');
            let waitingOverlay = document.getElementById('waitingOverlay');
            let statusDot = document.getElementById('statusDot');
            let statusText = document.getElementById('statusText');
            let hudSource = document.getElementById('hudSource');
            let hudFps = document.getElementById('hudFps');
            let hudRes = document.getElementById('hudRes');

            function switchTab(tab) {{
                document.getElementById('panelWired').classList.toggle('active', tab === 'wired');
                document.getElementById('panelNet').classList.toggle('active', tab === 'net');
                document.getElementById('panelMobile').classList.toggle('active', tab === 'mobile');

                document.getElementById('tabBtnWired').classList.toggle('active', tab === 'wired');
                document.getElementById('tabBtnNet').classList.toggle('active', tab === 'net');
                document.getElementById('tabBtnMobile').classList.toggle('active', tab === 'mobile');
            }}

            function setPreset(url) {{
                document.getElementById('streamUri').value = url;
            }}

            async function setOrientation(mode) {{
                document.querySelectorAll('.btn-rot').forEach(b => b.classList.remove('active'));
                const btn = document.getElementById('rot-' + (mode === 'rotate_90_cw' ? 'cw' : mode === 'rotate_90_ccw' ? 'ccw' : mode === 'rotate_180' ? '180' : mode === 'flip_h' ? 'fliph' : 'none'));
                if (btn) btn.classList.add('active');

                try {{
                    await fetch('/camera/transform?transform=' + encodeURIComponent(mode), {{ method: 'POST' }});
                }} catch(e) {{}}
            }}

            async function detectUsbCams() {{
                const log = document.getElementById('usbStatusLog');
                const select = document.getElementById('usbCameraSelect');
                log.textContent = 'Scanning USB & video devices...';
                log.style.color = '#38bdf8';

                try {{
                    const res = await fetch('/usb/cameras');
                    const data = await res.json();
                    select.innerHTML = '';

                    if (!data.cameras || data.cameras.length === 0) {{
                        log.textContent = '⚠️ No USB cameras detected. Please check your USB cable connection.';
                        log.style.color = '#f59e0b';
                        const opt = document.createElement('option');
                        opt.value = '0';
                        opt.textContent = '/dev/video0 (No devices detected)';
                        select.appendChild(opt);
                        return;
                    }}

                    data.cameras.forEach(c => {{
                        const opt = document.createElement('option');
                        opt.value = c.device_index;
                        const badge = c.mobile_candidate ? ' 📱 [Mobile/External USB]' : (c.internal ? ' 💻 [Laptop Built-in]' : '');
                        opt.textContent = `${{c.id}} — ${{c.name}}${{badge}}`;
                        if (c.mobile_candidate && !select.value) {{
                            opt.selected = true;
                        }}
                        select.appendChild(opt);
                    }});

                    log.textContent = `✅ Found ${{data.cameras.length}} video device(s). Select device and click Start.`;
                    log.style.color = '#4ade80';
                }} catch(e) {{
                    log.textContent = '❌ Error detecting cameras: ' + e.message;
                    log.style.color = '#ef4444';
                }}
            }}

            async function startUsbCam() {{
                const select = document.getElementById('usbCameraSelect');
                const idx = select.value;
                const log = document.getElementById('usbStatusLog');
                log.textContent = `Starting /dev/video${{idx}}...`;
                log.style.color = '#38bdf8';

                try {{
                    const res = await fetch(`/usb/start?device_index=${{idx}}&fps=25`, {{ method: 'POST' }});
                    const data = await res.json();
                    if (res.ok) {{
                        log.textContent = `✅ Live stream active on /dev/video${{idx}}`;
                        log.style.color = '#4ade80';
                        // Force refresh stream image
                        document.getElementById('liveStream').src = '/stream/detect?t=' + Date.now();
                    }} else {{
                        log.textContent = '❌ ' + (data.detail || 'Failed to start camera');
                        log.style.color = '#ef4444';
                    }}
                }} catch(e) {{
                    log.textContent = '❌ Error: ' + e.message;
                    log.style.color = '#ef4444';
                }}
            }}

            async function connectIpCam() {{
                const uri = document.getElementById('streamUri').value;
                const log = document.getElementById('netStatusLog');
                log.textContent = `Connecting to ${{uri}}...`;
                log.style.color = '#38bdf8';

                try {{
                    const res = await fetch(`/camera/connect?source_uri=${{encodeURIComponent(uri)}}&fps=20`, {{ method: 'POST' }});
                    const data = await res.json();
                    if (res.ok) {{
                        log.textContent = `✅ Connected to ${{uri}}`;
                        log.style.color = '#4ade80';
                        document.getElementById('liveStream').src = '/stream/detect?t=' + Date.now();
                    }} else {{
                        log.textContent = '❌ ' + (data.detail || 'Connection failed');
                        log.style.color = '#ef4444';
                    }}
                }} catch(e) {{
                    log.textContent = '❌ Error: ' + e.message;
                    log.style.color = '#ef4444';
                }}
            }}

            async function stopCamera() {{
                try {{
                    await fetch('/camera/disconnect', {{ method: 'POST' }});
                    document.getElementById('usbStatusLog').textContent = '⏹️ Camera disconnected';
                    document.getElementById('usbStatusLog').style.color = '#94a3b8';
                    document.getElementById('netStatusLog').textContent = '⏹️ Camera disconnected';
                    document.getElementById('netStatusLog').style.color = '#94a3b8';
                }} catch(e) {{}}
            }}

            async function pollStatus() {{
                try {{
                    let res = await fetch('/stream/status');
                    let data = await res.json();
                    let health = data.camera_health || {{}};

                    if (data.ingest_active) {{
                        waitingOverlay.classList.add('hidden');
                        statusDot.classList.add('live');
                        statusText.textContent = 'Live (' + (health.medium || 'Active') + ') — ' + (health.fps || 0) + ' FPS';
                        
                        hudSource.textContent = health.medium || 'Live Camera';
                        hudFps.textContent = (health.fps || 0) + ' FPS';
                        hudRes.textContent = health.resolution || 'N/A';
                    }} else {{
                        waitingOverlay.classList.remove('hidden');
                        statusDot.classList.remove('live');
                        statusText.textContent = health.connecting ? 'Connecting...' : 'Waiting for Camera...';
                        hudSource.textContent = 'Disconnected';
                        hudFps.textContent = '0.0 FPS';
                        hudRes.textContent = '0x0';
                    }}

                    // Orientation button sync
                    if (data.transform) {{
                        const curMode = data.transform;
                        document.querySelectorAll('.btn-rot').forEach(b => b.classList.remove('active'));
                        const btn = document.getElementById('rot-' + (curMode === 'rotate_90_cw' ? 'cw' : curMode === 'rotate_90_ccw' ? 'ccw' : curMode === 'rotate_180' ? '180' : curMode === 'flip_h' ? 'fliph' : 'none'));
                        if (btn) btn.classList.add('active');
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
            detectUsbCams();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    http_port = int(os.getenv("PORT", "8002"))
    print(f"Starting Helmet Detection HTTP server on http://0.0.0.0:{http_port}")
    uvicorn.run(app, host="0.0.0.0", port=http_port, log_level="info")
