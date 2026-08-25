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
from streamer import FrameBuffer, mjpeg_from_buffer

load_dotenv(find_dotenv(usecwd=True))

app = FastAPI(
    title="Helmet Detection",
    description="Detect persons with/without helmets from live camera frames",
    version="1.1.0",
)

_detector: Optional[HelmetDetector] = None
_ingest_buffer = FrameBuffer(max_frames=10)
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


@app.on_event("startup")
async def startup():
    try:
        _get_detector()
    except Exception as e:
        print(f"Warning: detector init failed: {e}")

    t = threading.Thread(target=_async_detection_worker, daemon=True)
    t.start()
    print("Async background AI helmet detection worker started.")


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
    with _detection_lock:
        info = dict(_last_detection_info)
    return {
        "ingest_active": _ingest_buffer.is_active,
        "ingest_frames": _ingest_buffer.frame_count,
        "latest_detection": info,
        "local_ip": local_ip,
        "port": port,
    }


@app.get("/mobile", response_class=HTMLResponse)
async def mobile_page():
    local_ip = _get_local_ip()
    port = int(os.getenv("PORT", "8002"))
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
            .camera-container {{ position: relative; width: 100%; max-width: 460px; margin: 0 auto 12px; border-radius: 16px; overflow: hidden; background: #1e293b; border: 2px solid #334155; aspect-ratio: 4/3; }}
            video {{ width: 100%; height: 100%; object-fit: cover; }}
            canvas {{ display: none; }}
            .stats-badge {{ position: absolute; top: 10px; left: 10px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(6px); border: 1px solid #38bdf8; border-radius: 20px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600; color: #38bdf8; display: flex; align-items: center; gap: 6px; }}
            .live-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #38bdf8; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.4; transform: scale(0.85); }} }}
            .controls {{ display: flex; flex-direction: column; gap: 10px; max-width: 460px; margin: 0 auto; }}
            .btn {{ width: 100%; padding: 14px; border: none; border-radius: 12px; font-size: 1rem; font-weight: bold; cursor: pointer; transition: all 0.2s; }}
            .btn-start {{ background: linear-gradient(135deg, #0284c7, #0369a1); color: white; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3); }}
            .btn-snap {{ background: linear-gradient(135deg, #22c55e, #16a34a); color: white; box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3); }}
            .btn-stop {{ background: #ef4444; color: white; display: none; }}
            .btn-switch {{ background: #334155; color: #e2e8f0; }}
            .status-box {{ background: #1e293b; border-radius: 12px; padding: 14px; margin-top: 12px; max-width: 460px; margin-left: auto; margin-right: auto; text-align: left; font-size: 0.85rem; border: 1px solid #334155; }}
            .status-row {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
            .status-label {{ color: #94a3b8; }}
            .status-value {{ font-weight: bold; color: #e2e8f0; }}
            .snap-preview {{ max-width: 100%; border-radius: 8px; margin-top: 10px; display: none; }}
            .alert-pill {{ padding: 6px 12px; border-radius: 8px; font-weight: bold; margin-top: 8px; text-align: center; }}
            .alert-safe {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }}
            .alert-violation {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⛑️ Helmet Detection Stream</h1>
            <p>Live AI Safety Monitoring</p>
        </div>

        <div class="camera-container">
            <video id="video" autoplay playsinline muted></video>
            <img id="snapPreview" class="snap-preview" alt="Captured Frame">
            <canvas id="canvas"></canvas>
            <div class="stats-badge" id="liveBadge" style="display: none;">
                <span class="live-dot"></span> <span id="badgeText">STREAMING TO SERVER</span>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-start" id="startBtn" onclick="startCamera()">📹 Start Live Video Stream</button>
            <button class="btn btn-stop" id="stopBtn" onclick="stopCamera()">⏹️ Stop Stream</button>
            <button class="btn btn-switch" id="switchBtn" onclick="switchCamera()">🔄 Switch Camera (Front/Back)</button>
            <button class="btn btn-snap" onclick="document.getElementById('nativeCamInput').click()">📸 Snap Photo from Camera</button>
            <input type="file" id="nativeCamInput" accept="image/*" capture="environment" style="display: none;" onchange="handleNativeSnap(event)">
        </div>

        <div class="status-box">
            <div class="status-row">
                <span class="status-label">Status:</span>
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

            async function startCamera() {{
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
                    document.getElementById('streamStatus').style.color = '#38bdf8';

                    if (streamInterval) clearInterval(streamInterval);
                    streamInterval = setInterval(sendFrame, 40);
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
                            updateStats(data);
                        }})
                        .catch(err => {{ isSending = false; console.error('Frame error', err); }});
                }}, 'image/jpeg', 0.65);
            }}

            function updateStats(data) {{
                let p = data.num_persons || 0;
                let v = data.num_violations || 0;
                let s = data.num_helmets || 0;
                document.getElementById('personCount').textContent = p;
                let safetyElem = document.getElementById('safetyStatus');
                if (p === 0) {{
                    safetyElem.textContent = 'No Person';
                    safetyElem.style.color = '#94a3b8';
                }} else if (v > 0) {{
                    safetyElem.textContent = '⚠️ ' + v + ' NO HELMET VIOLATION';
                    safetyElem.style.color = '#ef4444';
                }} else {{
                    safetyElem.textContent = '✅ ' + s + ' HELMET OK (SAFE)';
                    safetyElem.style.color = '#22c55e';
                }}
            }}

            function handleNativeSnap(event) {{
                const file = event.target.files[0];
                if (!file) return;
                stopCamera();
                video.style.display = 'none';
                snapPreview.src = URL.createObjectURL(file);
                snapPreview.style.display = 'block';

                document.getElementById('streamStatus').textContent = 'Analyzing photo...';
                document.getElementById('streamStatus').style.color = '#38bdf8';

                let fd = new FormData();
                fd.append('file', file);
                fetch('/ingest/frame', {{ method: 'POST', body: fd }})
                    .then(r => r.json())
                    .then(data => {{
                        frameCount++;
                        document.getElementById('framesSent').textContent = frameCount;
                        document.getElementById('streamStatus').textContent = 'Processed';
                        document.getElementById('streamStatus').style.color = '#38bdf8';
                        updateStats(data);
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

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8002")))

