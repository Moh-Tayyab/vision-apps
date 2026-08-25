"""FastAPI app for Carton Counter - pallet counting with YOLO detection."""

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

import os
import tempfile
import time
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response

from detector import CartonDetector
from counter import CartonCounter, CountResult
from streamer import FrameBuffer, MobileCameraStream, mjpeg_from_buffer, websocket_stream
import threading

app = FastAPI(
    title="Carton Counter",
    description="Pallet carton counting system with multi-angle fusion",
    version="1.1.0",
)

_detector: Optional[CartonDetector] = None
_counter: Optional[CartonCounter] = None
_stream: Optional[MobileCameraStream] = None
_ingest_buffer = FrameBuffer(max_frames=10)
_last_detection_info = {
    "count": 0,
    "detections": [],
    "inference_time_ms": 0.0,
    "timestamp": 0.0,
}
_cached_detections: list = []
_detection_lock = threading.Lock()
_worker_running = True


def _async_detection_worker():
    """Continuous background worker that runs AI inference on fresh video frames."""
    global _cached_detections, _last_detection_info
    last_seen_ts = 0.0
    while _worker_running:
        try:
            buf = _ingest_buffer if _ingest_buffer.is_active else (_stream._buffer if _stream and _stream.is_active else None)
            if buf is None:
                time.sleep(0.04)
                continue

            latest = buf.get_latest()
            if latest is None:
                time.sleep(0.04)
                continue

            ts, frame = latest
            if ts != last_seen_ts:
                last_seen_ts = ts
                conf = float(os.getenv("CONF_THRESHOLD", "0.36"))
                result = _get_detector().detect(frame, confidence=conf)
                with _detection_lock:
                    _cached_detections = result.detections
                    _last_detection_info = {
                        "count": len(result.detections),
                        "detections": [d.to_dict() for d in result.detections],
                        "inference_time_ms": result.inference_time_ms,
                        "timestamp": time.time(),
                    }
            else:
                time.sleep(0.02)
        except Exception as e:
            time.sleep(0.05)


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


def _get_detector() -> CartonDetector:
    global _detector
    if _detector is None:
        _detector = CartonDetector()
    return _detector


def _get_counter() -> CartonCounter:
    global _counter
    if _counter is None:
        _counter = CartonCounter(_get_detector())
    return _counter


def _get_stream() -> MobileCameraStream:
    global _stream
    if _stream is None:
        source = os.getenv("VIDEO_SOURCE", "0")
        try:
            source = int(source)
        except ValueError:
            pass
        fps = int(os.getenv("STREAM_FPS", "30"))
        _stream = MobileCameraStream(source=source, fps=fps)
    return _stream


def _read_image(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    return img


@app.on_event("startup")
async def startup():
    try:
        _get_detector()
        _generate_demo_images()
    except Exception as e:
        print(f"Warning: detector init failed: {e}")
    # Start async AI worker thread for real-time zero-lag stream
    t = threading.Thread(target=_async_detection_worker, daemon=True)
    t.start()
    print("Async background AI detection worker started.")


def _generate_demo_images():
    """Generate demo images on startup if they don't exist."""
    demo_dir = os.path.join(os.path.dirname(__file__), "demo_images")
    if not os.path.isdir(demo_dir):
        try:
            from demo_images import generate_carton_image, generate_multi_view_images
            os.makedirs(demo_dir, exist_ok=True)
            front, side, top = generate_multi_view_images(12)
            cv2.imwrite(os.path.join(demo_dir, "front.jpg"), front)
            cv2.imwrite(os.path.join(demo_dir, "side.jpg"), side)
            cv2.imwrite(os.path.join(demo_dir, "top.jpg"), top)
            single = generate_carton_image(8, seed=99)
            cv2.imwrite(os.path.join(demo_dir, "single.jpg"), single)
            print(f"Demo images generated in {demo_dir}")
        except Exception as e:
            print(f"Warning: could not generate demo images: {e}")


@app.get("/demo-images/{filename}")
async def serve_demo_image(filename: str):
    """Serve demo images for the UI."""
    demo_dir = os.path.join(os.path.dirname(__file__), "demo_images")
    filepath = os.path.join(demo_dir, filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Demo image not found")
    import mimetypes
    media_type = mimetypes.guess_type(filepath)[0] or "image/jpeg"
    with open(filepath, "rb") as f:
        content = f.read()
    return Response(content=content, media_type=media_type)


@app.on_event("shutdown")
async def shutdown():
    global _stream
    if _stream is not None:
        _stream.stop()
        _stream = None


@app.get("/health")
async def health():
    detector = _get_detector()
    stream = _get_stream()
    return {
        "status": "healthy",
        "backend": detector.backend,
        "stream_active": stream.is_active,
        "ingest_active": _ingest_buffer.is_active,
        "ingest_frames": _ingest_buffer.frame_count,
    }


@app.get("/model/info")
async def model_info():
    detector = _get_detector()
    return detector.get_model_info()


@app.post("/detect")
async def detect(file: UploadFile = File(...), confidence: float = Query(default=0.5)):
    contents = await file.read()
    image = _read_image(contents)

    detector = _get_detector()
    result = detector.detect(image, confidence=confidence)
    return result.to_dict()


def _draw_detections(image: np.ndarray, detections) -> np.ndarray:
    vis = image.copy()
    h, w = vis.shape[:2]
    count = len(detections)
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y1 = max(0, y1 - th - 6)
        label_y2 = y1 if y1 >= th + 6 else y1 + th + 6
        cv2.rectangle(vis, (x1, label_y1), (x1 + tw + 6, label_y2), (0, 200, 0), -1)
        cv2.putText(vis, label, (x1 + 3, label_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Overlay Top Header Banner with Total Carton Count
    overlay = vis.copy()
    banner_w = min(280, w - 20)
    cv2.rectangle(overlay, (10, 10), (10 + banner_w, 55), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
    cv2.rectangle(vis, (10, 10), (10 + banner_w, 55), (34, 197, 94), 2)
    cv2.putText(vis, f"CARTONS: {count}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (74, 222, 128), 2, cv2.LINE_AA)
    return vis


@app.post("/detect/visualize")
async def detect_visualize(
    file: UploadFile = File(...),
    confidence: Optional[float] = Query(default=None),
):
    contents = await file.read()
    image = _read_image(contents)

    conf = confidence if confidence is not None else float(os.getenv("CONF_THRESHOLD", "0.36"))
    detector = _get_detector()
    result = detector.detect(image, confidence=conf)

    vis = _draw_detections(image, result.detections)
    _, buffer = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/count")
async def count(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    top: UploadFile = File(...),
    confidence: Optional[float] = Query(default=None),
):
    counter = _get_counter()
    images = []
    for f in [front, side, top]:
        contents = await f.read()
        images.append(_read_image(contents))

    result = counter.count_multi_angle(images)
    return result.to_dict()


@app.post("/pallet/angle")
async def pallet_angle(file: UploadFile = File(...)):
    """Estimate pallet 3D orientation (pitch/roll/yaw) from one view."""
    contents = await file.read()
    image = _read_image(contents)

    counter = _get_counter()
    angle = counter.detect_pallet_angle(image)
    return angle.to_dict()


@app.post("/pallet/correct")
async def pallet_correct(file: UploadFile = File(...)):
    """Return the perspective-corrected image for the detected pallet plane."""
    contents = await file.read()
    image = _read_image(contents)

    counter = _get_counter()
    det_result = counter.detector.detect(image)
    angle = counter.angle_detector.detect_angle(image, det_result.detections)
    corrected = counter.angle_detector.apply_perspective_correction(image, angle)
    if corrected is None:
        raise HTTPException(status_code=422, detail="Cannot estimate pallet plane (need >= 4 visible cartons)")
    _, buffer = cv2.imencode(".jpg", corrected, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/count/video")
async def count_video(
    file: UploadFile = File(...),
    sample_frames: int = Query(default=5, ge=1, le=30),
):
    contents = await file.read()
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(contents)
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise HTTPException(status_code=400, detail="Empty video")

        indices = np.linspace(0, total - 1, min(sample_frames, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
    finally:
        os.unlink(tmp.name)

    counter = _get_counter()
    result = counter.count_multi_frame(frames)
    return result.to_dict()


@app.post("/ingest/frame")
async def ingest_frame(file: UploadFile = File(...)):
    """Accept a frame pushed from any camera client immediately without latency."""
    image = _read_image(await file.read())
    _ingest_buffer.update(image)

    with _detection_lock:
        current_count = _last_detection_info.get("count", 0)

    return {
        "status": "accepted",
        "ingest_frames": _ingest_buffer.frame_count,
        "count": current_count,
        "size": [image.shape[1], image.shape[0]],
    }


def _live_source() -> str:
    return "ingest" if _ingest_buffer.is_active else "local_camera"


@app.get("/stream")
async def stream():
    """MJPEG live view. Serves pushed frames when available, else local camera."""
    if _ingest_buffer.is_active:
        gen = mjpeg_from_buffer(_ingest_buffer, fps_limit=30.0)
    else:
        gen = _ensure_started(_get_stream()).mjpeg_generator()
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/detect")
async def stream_detect(confidence: Optional[float] = Query(default=None)):
    """High-speed 30 FPS MJPEG stream with async bounding boxes overlay."""
    def annotate(frame: np.ndarray) -> np.ndarray:
        with _detection_lock:
            current_dets = list(_cached_detections)
        return _draw_detections(frame, current_dets)

    if _ingest_buffer.is_active:
        gen = mjpeg_from_buffer(_ingest_buffer, quality=75, transform=annotate, fps_limit=30.0)
    else:
        stream_obj = _ensure_started(_get_stream())
        buffer = stream_obj._buffer
        gen = mjpeg_from_buffer(buffer, quality=75, transform=annotate, fps_limit=30.0)
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/status")
async def stream_status():
    stream_obj = _stream
    return {
        "ingest_active": _ingest_buffer.is_active,
        "ingest_frames": _ingest_buffer.frame_count,
        "stream_active": stream_obj.is_active if stream_obj else False,
        "latest_detection": _last_detection_info,
        "local_ip": _get_local_ip(),
        "port": int(os.getenv("PORT", "8001")),
    }


def _ensure_started(stream_obj: MobileCameraStream) -> MobileCameraStream:
    if not stream_obj.is_active:
        stream_obj.start()
    return stream_obj


@app.get("/stream/start")
async def stream_start():
    stream_obj = _get_stream()
    stream_obj.start()
    return {"status": "started", "source": stream_obj.source}


@app.get("/stream/stop")
async def stream_stop():
    global _stream
    if _stream is not None:
        _stream.stop()
        _stream = None
    return {"status": "stopped"}


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    stream_obj = _get_stream()
    if not stream_obj.is_active:
        stream_obj.start()
    await websocket_stream(websocket, stream_obj)


@app.get("/mobile", response_class=HTMLResponse)
async def mobile_camera_page():
    local_ip = _get_local_ip()
    port = int(os.getenv("PORT", "8001"))
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>Carton Counter - Mobile Stream</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 15px; text-align: center; }}
            .header {{ margin-bottom: 12px; }}
            h1 {{ font-size: 1.4rem; color: #60a5fa; margin-bottom: 4px; }}
            p {{ font-size: 0.85rem; color: #94a3b8; }}
            .https-banner {{ background: #7c2d12; border: 1px solid #ea580c; border-radius: 12px; padding: 12px; margin-bottom: 15px; text-align: left; font-size: 0.85rem; }}
            .https-banner h3 {{ color: #fdba74; font-size: 0.95rem; margin-bottom: 4px; }}
            .camera-container {{ position: relative; width: 100%; max-width: 460px; margin: 0 auto 12px; border-radius: 16px; overflow: hidden; background: #1e293b; border: 2px solid #334155; aspect-ratio: 4/3; }}
            video {{ width: 100%; height: 100%; object-fit: cover; }}
            canvas {{ display: none; }}
            .stats-badge {{ position: absolute; top: 10px; left: 10px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(6px); border: 1px solid #4ade80; border-radius: 20px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600; color: #4ade80; display: flex; align-items: center; gap: 6px; }}
            .live-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #4ade80; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.4; transform: scale(0.85); }} }}
            .controls {{ display: flex; flex-direction: column; gap: 10px; max-width: 460px; margin: 0 auto; }}
            .btn {{ width: 100%; padding: 14px; border: none; border-radius: 12px; font-size: 1rem; font-weight: bold; cursor: pointer; transition: all 0.2s; }}
            .btn-start {{ background: linear-gradient(135deg, #22c55e, #16a34a); color: white; box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3); }}
            .btn-snap {{ background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); }}
            .btn-stop {{ background: #ef4444; color: white; display: none; }}
            .btn-switch {{ background: #334155; color: #e2e8f0; }}
            .btn-https {{ background: #ea580c; color: white; margin-top: 6px; }}
            .status-box {{ background: #1e293b; border-radius: 12px; padding: 14px; margin-top: 12px; max-width: 460px; margin-left: auto; margin-right: auto; text-align: left; font-size: 0.85rem; border: 1px solid #334155; }}
            .status-row {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
            .status-label {{ color: #94a3b8; }}
            .status-value {{ font-weight: bold; color: #e2e8f0; }}
            .links {{ margin-top: 15px; display: flex; flex-direction: column; gap: 8px; }}
            .link-btn {{ color: #60a5fa; text-decoration: none; font-size: 0.9rem; font-weight: 500; }}
            .snap-preview {{ max-width: 100%; border-radius: 8px; margin-top: 10px; display: none; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📱 Mobile Camera Stream</h1>
            <p>Live Carton Detection & Counting</p>
        </div>

        <div id="httpsBanner" class="https-banner" style="display: none;">
            <h3>🔒 Live Video requires HTTPS</h3>
            <p>Mobile browsers only allow live video streaming over HTTPS. Tap below to switch to HTTPS (if prompted, tap <i>Advanced &rarr; Proceed</i>):</p>
            <button class="btn btn-https" onclick="switchToHttps()">🔒 Switch to HTTPS Stream</button>
        </div>

        <div class="camera-container">
            <video id="video" autoplay playsinline muted></video>
            <img id="snapPreview" class="snap-preview" alt="Captured Frame">
            <canvas id="canvas"></canvas>
            <div class="stats-badge" id="liveBadge" style="display: none;">
                <span class="live-dot"></span> <span id="badgeText">STREAMING TO LAPTOP</span>
            </div>
        </div>

        <div class="controls">
            <!-- Mode 1: Live Video Stream (getUserMedia) -->
            <button class="btn btn-start" id="startBtn" onclick="startCamera()">📹 Start Live Video Stream</button>
            <button class="btn btn-stop" id="stopBtn" onclick="stopCamera()">⏹️ Stop Stream</button>
            <button class="btn btn-switch" id="switchBtn" onclick="switchCamera()">🔄 Switch Camera (Front/Back)</button>

            <!-- Mode 2: Direct Photo Capture (works on HTTP & HTTPS) -->
            <button class="btn btn-snap" onclick="document.getElementById('nativeCamInput').click()">📸 Snap Photo from Camera & Send</button>
            <input type="file" id="nativeCamInput" accept="image/*" capture="environment" style="display: none;" onchange="handleNativeSnap(event)">
        </div>

        <div class="status-box">
            <div class="status-row">
                <span class="status-label">Stream Status:</span>
                <span class="status-value" id="streamStatus" style="color: #94a3b8;">Ready</span>
            </div>
            <div class="status-row">
                <span class="status-label">Frames Sent to AI:</span>
                <span class="status-value" id="framesSent">0</span>
            </div>
            <div class="status-row">
                <span class="status-label">Stream Speed:</span>
                <span class="status-value" id="fpsRate">0 fps</span>
            </div>
            <div class="status-row">
                <span class="status-label">Server IP:</span>
                <span class="status-value">{local_ip}:{port}</span>
            </div>
        </div>

        <div class="links">
            <a class="link-btn" href="/" target="_blank">💻 View Full Dashboard on Laptop</a>
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
                window.location.href = 'https://' + window.location.hostname + ':' + (window.location.port || '8001') + '/mobile';
            }}

            async function startCamera() {{
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
                    document.getElementById('httpsBanner').style.display = 'block';
                    alert("Mobile browser requires HTTPS for live video stream.\\n\\nSwitching to HTTPS now or use '📸 Snap Photo from Camera' button below.");
                    switchToHttps();
                    return;
                }}
                try {{
                    if (stream) {{
                        stream.getTracks().forEach(t => t.stop());
                    }}
                    stream = await navigator.mediaDevices.getUserMedia({{
                        video: {{
                            facingMode: {{ ideal: facingMode }},
                            width: {{ ideal: 1280 }},
                            height: {{ ideal: 720 }}
                        }},
                        audio: false
                    }});
                    video.style.display = 'block';
                    snapPreview.style.display = 'none';
                    video.srcObject = stream;
                    await video.play();

                    document.getElementById('startBtn').style.display = 'none';
                    document.getElementById('stopBtn').style.display = 'block';
                    document.getElementById('liveBadge').style.display = 'flex';
                    document.getElementById('streamStatus').textContent = 'Streaming Live';
                    document.getElementById('streamStatus').style.color = '#4ade80';

                    if (streamInterval) clearInterval(streamInterval);
                    streamInterval = setInterval(sendFrame, 50);
                }} catch(e) {{
                    alert('Camera Error: ' + e.message);
                    document.getElementById('streamStatus').textContent = 'Error: ' + e.message;
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
                document.getElementById('startBtn').style.display = 'block';
                document.getElementById('stopBtn').style.display = 'none';
                document.getElementById('liveBadge').style.display = 'none';
                document.getElementById('streamStatus').textContent = 'Stopped';
                document.getElementById('streamStatus').style.color = '#94a3b8';
            }}

            async function switchCamera() {{
                facingMode = (facingMode === 'environment') ? 'user' : 'environment';
                if (stream) {{
                    await startCamera();
                }}
            }}

            let isSending = false;
            function sendFrame() {{
                if (!video.videoWidth || isSending) return;
                canvas.width = Math.min(video.videoWidth, 640);
                canvas.height = Math.min(video.videoHeight, 480);
                let ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
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
                            if (data.count !== undefined) {{
                                document.getElementById('streamStatus').textContent = 'Cartons Detected: ' + data.count;
                                document.getElementById('streamStatus').style.color = '#4ade80';
                            }}
                            let now = Date.now();
                            let elapsed = (now - lastFrameTime) / 1000;
                            if (elapsed > 0) {{
                                document.getElementById('fpsRate').textContent = (1 / elapsed).toFixed(1) + ' fps';
                            }}
                            lastFrameTime = now;
                        }})
                        .catch(err => {{ isSending = false; console.error('Frame send error', err); }});
                }}, 'image/jpeg', 0.70);
            }}

            function handleNativeSnap(event) {{
                const file = event.target.files[0];
                if (!file) return;
                stopCamera();
                video.style.display = 'none';
                snapPreview.src = URL.createObjectURL(file);
                snapPreview.style.display = 'block';

                document.getElementById('streamStatus').textContent = 'Analyzing photo...';
                document.getElementById('streamStatus').style.color = '#60a5fa';

                let fd = new FormData();
                fd.append('file', file);
                fetch('/ingest/frame', {{ method: 'POST', body: fd }})
                    .then(r => r.json())
                    .then(data => {{
                        frameCount++;
                        document.getElementById('framesSent').textContent = frameCount;
                        document.getElementById('streamStatus').textContent = 'Cartons Detected: ' + (data.count !== undefined ? data.count : 'Done');
                        document.getElementById('streamStatus').style.color = '#4ade80';
                    }})
                    .catch(err => {{
                        document.getElementById('streamStatus').textContent = 'Send failed: ' + err.message;
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
    port = int(os.getenv("PORT", "8001"))
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Carton Counter — Live Monitor</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: #000; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; font-family: Arial, sans-serif; }}

            /* ── TOP BAR ── */
            .topbar {{
                width: 100%;
                background: #111;
                padding: 12px 24px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .topbar-title {{ color: #fff; font-size: 1.1rem; font-weight: bold; }}
            .status-dot {{ width: 12px; height: 12px; border-radius: 50%; background: #ef4444; display: inline-block; margin-right: 8px; }}
            .status-dot.live {{ background: #22c55e; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:0.3 }} }}
            .status-label {{ color: #94a3b8; font-size: 0.9rem; }}

            /* ── MAIN VIDEO ── */
            .video-wrapper {{
                position: relative;
                width: 100%;
                max-width: 960px;
                background: #111;
            }}
            #liveStream {{
                width: 100%;
                display: block;
                min-height: 480px;
                object-fit: contain;
                background: #111;
            }}
            /* waiting overlay */
            .waiting-overlay {{
                position: absolute;
                inset: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background: #111;
                color: #475569;
                font-size: 1.1rem;
                gap: 16px;
            }}
            .waiting-overlay.hidden {{ display: none; }}
            .spinner {{ width: 48px; height: 48px; border: 4px solid #334155; border-top-color: #60a5fa; border-radius: 50%; animation: spin 1s linear infinite; }}
            @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

            /* ── COUNT BADGE ── */
            .count-badge {{
                background: #111;
                padding: 16px 32px;
                text-align: center;
                width: 100%;
                max-width: 960px;
                border-top: 1px solid #1e293b;
            }}
            .count-number {{ font-size: 4rem; font-weight: 900; color: #4ade80; line-height: 1; }}
            .count-label {{ color: #94a3b8; font-size: 0.95rem; margin-top: 4px; letter-spacing: 2px; text-transform: uppercase; }}

            /* ── MOBILE HINT ── */
            .mobile-hint {{
                width: 100%;
                max-width: 960px;
                background: #0f172a;
                border-top: 1px solid #1e293b;
                padding: 14px 24px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .mobile-hint span {{ color: #64748b; font-size: 0.85rem; }}
            .mobile-link {{ color: #38bdf8; font-size: 0.85rem; font-weight: bold; text-decoration: none; word-break: break-all; }}
        </style>
    </head>
    <body>

        <!-- TOP BAR -->
        <div class="topbar">
            <span class="topbar-title">📦 Carton Counter — Live Monitor</span>
            <span>
                <span class="status-dot" id="statusDot"></span>
                <span class="status-label" id="statusLabel">Waiting for mobile camera...</span>
            </span>
        </div>

        <!-- VIDEO STREAM -->
        <div class="video-wrapper">
            <img id="liveStream" src="/stream/detect" alt="live stream"
                 onerror="this.src='/stream/detect?t='+Date.now()"
                 onload="streamLoaded()">
            <div class="waiting-overlay" id="waitingOverlay">
                <div class="spinner"></div>
                <span>Mobile camera connect hone ka intezaar hai…</span>
                <small style="color:#334155">Mobile par link kholein aur camera start karein</small>
            </div>
        </div>

        <!-- COUNT -->
        <div class="count-badge">
            <div class="count-number" id="countNum">—</div>
            <div class="count-label">Cartons Detected</div>
        </div>

        <!-- MOBILE HINT -->
        <div class="mobile-hint">
            <span>📱 Mobile link:</span>
            <a class="mobile-link" href="https://pvc-tiles-homepage-hydrogen.trycloudflare.com/mobile" target="_blank">
                https://pvc-tiles-homepage-hydrogen.trycloudflare.com/mobile
            </a>
        </div>

        <script>
            let streamActive = false;

            function streamLoaded() {{
                document.getElementById('waitingOverlay').classList.add('hidden');
            }}

            // Poll /stream/status every 1.5 s to update count and status dot
            async function pollStatus() {{
                try {{
                    const res = await fetch('/stream/status');
                    const data = await res.json();
                    const ingestActive = data.ingest_active;
                    const count = data.latest_detection ? data.latest_detection.count : 0;
                    const ts = data.latest_detection ? data.latest_detection.timestamp : 0;
                    const fresh = ts && (Date.now()/1000 - ts) < 5;

                    // Update count
                    document.getElementById('countNum').textContent = fresh ? count : '—';

                    // Update status dot & label
                    const dot = document.getElementById('statusDot');
                    const label = document.getElementById('statusLabel');
                    if (ingestActive && fresh) {{
                        dot.classList.add('live');
                        label.textContent = 'LIVE — ' + data.ingest_frames + ' frames received';
                        document.getElementById('waitingOverlay').classList.add('hidden');
                    }} else {{
                        dot.classList.remove('live');
                        label.textContent = 'Waiting for mobile camera...';
                    }}
                }} catch(e) {{}}
            }}

            setInterval(pollStatus, 1500);
            pollStatus();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import asyncio
    import uvicorn

    async def _run_dual_servers():
        http_port = int(os.getenv("PORT", "8001"))
        https_port = int(os.getenv("HTTPS_PORT", "8443"))
        
        cert_file = os.path.join(os.path.dirname(__file__), "certs", "cert.pem")
        key_file = os.path.join(os.path.dirname(__file__), "certs", "key.pem")
        
        servers = []
        cfg_http = uvicorn.Config(app, host="0.0.0.0", port=http_port, log_level="info")
        servers.append(uvicorn.Server(cfg_http).serve())
        
        if os.path.exists(cert_file) and os.path.exists(key_file):
            cfg_https = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=https_port,
                ssl_keyfile=key_file,
                ssl_certfile=cert_file,
                log_level="info"
            )
            servers.append(uvicorn.Server(cfg_https).serve())
            print(f"Server started on HTTP: http://0.0.0.0:{http_port} and HTTPS: https://0.0.0.0:{https_port}")
        else:
            print(f"Server started on HTTP: http://0.0.0.0:{http_port}")
        
        await asyncio.gather(*servers)

    asyncio.run(_run_dual_servers())
