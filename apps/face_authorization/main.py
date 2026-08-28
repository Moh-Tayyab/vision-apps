"""FastAPI app: Authorized/Unauthorized Person Detection (App 3).

Enroll authorized persons' face embeddings (deepface), then compare faces
from live video frames (USB camera, Mobile IP, or Web push).
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections import deque
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from camera_manager import CameraHealth, CameraManager, list_system_cameras
from face_engine import COSINE_THRESHOLD, DETECTOR_BACKEND, FaceEngine
from streamer import FrameBuffer, MobileCameraStream, apply_transform

app = FastAPI(
    title="Face Authorization",
    description="Authorized vs unauthorized person detection using deepface embeddings",
    version="1.0.0",
)

DATA_DIR = os.getenv("DATA_DIR", "data")
STORE_PATH = os.path.join(DATA_DIR, "embeddings.json")

# Unified live-stream state (matches carton_counter App 1). When active, the
# MobileCameraStream buffer is the single source of truth for both display and
# inference so every app's "connect mobile" flow behaves identically.
_engine: Optional[FaceEngine] = None
_camera_manager: Optional[CameraManager] = None
_mobile_stream: Optional[MobileCameraStream] = None
_events_log: deque = deque(maxlen=300)

# Orientation correction applied at capture time so display + inference agree.
# none | flip_h | flip_v | rotate_90_cw | rotate_90_ccw | rotate_180
_CAMERA_TRANSFORM: str = os.getenv("CAMERA_TRANSFORM", "none")

# Detection Cache & Async Inference
_detection_lock = threading.Lock()
_latest_detections: list = []
_last_event_timestamps: dict[str, float] = {}  # for event debouncing
_inference_running = True
_inference_thread: Optional[threading.Thread] = None


def _get_engine() -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine(STORE_PATH)
    return _engine


def _get_camera_manager() -> CameraManager:
    global _camera_manager
    if _camera_manager is None:
        _camera_manager = CameraManager()
    return _camera_manager


def frame_transform(frame: np.ndarray) -> np.ndarray:
    """Apply the active orientation correction to a frame."""
    return apply_transform(frame, _CAMERA_TRANSFORM)


def _get_active_buffer() -> FrameBuffer:
    """Return the live FrameBuffer being used for display + inference.

    Prefers the unified MobileCameraStream (App 1 method) when it is running,
    otherwise falls back to the CameraManager source (http_mjpeg/usb/rtsp/mobile push).
    """
    if _mobile_stream is not None and _mobile_stream.is_active:
        return _mobile_stream.buffer
    return _get_camera_manager().camera.buffer


def _get_local_ip() -> str:
    """LAN IP discovery for mobile instructions.

    Prefers the LAN_IP env var (injected by docker-compose from the host) because
    the socket-trick inside a bridged container only returns the bridge IP
    (e.g. 172.19.0.x), which the phone cannot reach.
    """
    lan = os.getenv("LAN_IP", "").strip()
    if lan:
        return lan
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


def _verify_frame(image: np.ndarray, log_events: bool = True) -> dict:
    """Detect faces in the frame and match each against stored embeddings."""
    try:
        from deepface import DeepFace
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"deepface/tensorflow not available in this environment: {e}",
        )

    h, w = image.shape[:2]
    infer_img = image
    scale_x = 1.0
    scale_y = 1.0
    # Downscale for ultra-fast deepface inference
    if w > 480:
        infer_w = 480
        infer_h = int(h * (480 / w))
        infer_img = cv2.resize(image, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        scale_x = w / infer_w
        scale_y = h / infer_h

    try:
        faces = DeepFace.extract_faces(
            img_path=infer_img,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            color_face="bgr",
        )
    except Exception:
        faces = []

    results = []
    now = time.time()

    for i, face in enumerate(faces):
        facial_area = face.get("facial_area", {})
        x = int(facial_area.get("x", 0) * scale_x)
        y = int(facial_area.get("y", 0) * scale_y)
        fw = int(facial_area.get("w", 0) * scale_x)
        fh = int(facial_area.get("h", 0) * scale_y)
        confidence = float(face.get("confidence", 0.0))
        if fw < 30 or fh < 30 or confidence < 0.45:
            continue
        match = _get_engine().identify_face(face["face"])
        entry = {
            "bbox": [x, y, x + fw, y + fh],
            "confidence": round(confidence, 3),
        }
        if match is None:
            entry.update(status="unknown", reason="no enrolled persons")
        else:
            entry["matched_name"] = match["name"]
            entry["distance"] = match["distance"]
            entry["status"] = "authorized" if match["authorized"] else "unauthorized"
        results.append(entry)

        if log_events and entry["status"] in ("authorized", "unauthorized"):
            target_key = entry.get("matched_name", entry["status"])
            last_logged = _last_event_timestamps.get(target_key, 0.0)
            if now - last_logged > 5.0:
                _last_event_timestamps[target_key] = now
                _events_log.append({"timestamp": now, **entry})

    return {
        "num_faces": len(results),
        "faces": results,
        "any_unauthorized": any(f["status"] == "unauthorized" for f in results),
    }


def _async_inference_worker():
    """Background worker that continuously runs DeepFace on latest camera frame."""
    global _latest_detections
    cm = _get_camera_manager()

    while _inference_running:
        latest = _get_active_buffer().get_latest()
        if latest is None:
            time.sleep(0.05)
            continue

        ts, frame = latest
        # Only infer on fresh frames
        if time.time() - ts > 5.0:
            time.sleep(0.05)
            continue

        try:
            res = _verify_frame(frame, log_events=True)
            with _detection_lock:
                _latest_detections = res.get("faces", [])
        except Exception:
            pass

        time.sleep(0.03)


_https_started = False


def _start_https_server():
    global _https_started
    if _https_started:
        return
    _https_started = True

    cert_file = os.path.join(os.path.dirname(__file__), "certs", "cert.pem")
    key_file = os.path.join(os.path.dirname(__file__), "certs", "key.pem")
    https_port = int(os.getenv("HTTPS_PORT", "8445"))

    if not (os.path.exists(cert_file) and os.path.exists(key_file)):
        try:
            os.makedirs(os.path.dirname(cert_file), exist_ok=True)
            import subprocess
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_file, "-out", cert_file,
                "-days", "365", "-nodes", "-subj", "/CN=face-auth"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    if os.path.exists(cert_file) and os.path.exists(key_file):
        try:
            import uvicorn
            print(f"Starting Face Auth HTTPS server on https://0.0.0.0:{https_port}")
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
    global _inference_thread
    try:
        _get_engine()
        _get_camera_manager()
        _inference_thread = threading.Thread(
            target=_async_inference_worker,
            name="FaceAuthInferenceWorker",
            daemon=True,
        )
        _inference_thread.start()

        # Start HTTPS server in background thread so mobile browsers can access getUserMedia
        https_thread = threading.Thread(
            target=_start_https_server,
            name="FaceAuthHTTPSServer",
            daemon=True,
        )
        https_thread.start()
    except Exception as e:
        print(f"Warning: startup init failed: {e}")


@app.on_event("shutdown")
async def shutdown():
    global _inference_running
    _inference_running = False
    _get_camera_manager().stop()


@app.get("/network/info")
async def network_info():
    """LAN IP + shareable links to open /mobile on the phone."""
    ip = _get_local_ip()
    port = int(os.getenv("PORT", "8003"))
    https_port = int(os.getenv("HTTPS_PORT", "8445"))
    return {
        "local_ip": ip,
        "port": port,
        "https_port": https_port,
        "mobile_url": f"http://{ip}:{port}/mobile",
        "mobile_https_url": f"https://{ip}:{https_port}/mobile",
        "stream_url": f"http://{ip}:{port}/stream/detect",
    }


@app.get("/health")
async def health():
    engine = _get_engine()
    cam_health = _get_camera_manager().get_health()
    return {
        "status": "healthy",
        "model_loaded": engine.model_loaded,
        "enrolled_persons": len(engine.list_persons()),
        "camera": cam_health.to_dict(),
    }


@app.get("/model/info")
async def model_info():
    from face_engine import MODEL_NAME

    return {
        "library": "deepface",
        "model_name": MODEL_NAME,
        "detector_backend": DETECTOR_BACKEND,
        "cosine_threshold": COSINE_THRESHOLD,
        "persons": _get_engine().list_persons(),
    }


# ---------------- Enrollment ----------------


@app.post("/persons/enroll")
async def enroll(name: str = Form(...), files: List[UploadFile] = File(...)):
    """Save face embeddings for an authorized person (1+ images)."""
    if not name.strip():
        raise HTTPException(status_code=422, detail="Name must not be empty")
    images = [_read_image(await f.read()) for f in files]
    try:
        result = _get_engine().enroll(name.strip(), images)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "enrolled", **result}


@app.get("/persons")
async def list_persons():
    return {"persons": _get_engine().list_persons()}


@app.delete("/persons/{name}")
async def delete_person(name: str):
    if not _get_engine().remove(name):
        raise HTTPException(status_code=404, detail=f"Person not found: {name}")
    return {"status": "deleted", "name": name}


@app.get("/persons/{name}/photo")
async def get_person_photo(name: str):
    """Serve the enrollment photo for a person (JPEG)."""
    photo_path = _get_engine().get_person_photo_path(name)
    if photo_path is None:
        raise HTTPException(status_code=404, detail=f"No photo found for: {name}")
    return FileResponse(photo_path, media_type="image/jpeg", filename=f"{name}.jpg")


# ---------------- Camera Source & Ingestion ----------------


@app.get("/camera/devices")
async def get_camera_devices():
    """List auto-detected USB / V4L2 video devices on the system."""
    return {"devices": list_system_cameras()}


@app.get("/camera/health")
async def get_camera_health():
    """Get active camera health, status, and FPS."""
    return _get_camera_manager().get_health().to_dict()


@app.post("/camera/configure")
async def configure_camera(
    source_type: str = Form(..., description="usb | http_mjpeg | rtsp | mobile | video_file"),
    source_uri: str = Form(..., description="Device index (e.g. 0), /dev/video0, URL, or 'browser'"),
    target_fps: int = Form(15, ge=1, le=60),
):
    """Configure active camera stream source dynamically."""
    valid_types = ("usb", "http_mjpeg", "rtsp", "mobile", "video_file")
    if source_type.lower() not in valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid source_type '{source_type}'. Must be one of: {valid_types}",
        )
    health_info = _get_camera_manager().configure_camera(
        source_type=source_type,
        source_uri=source_uri,
        target_fps=target_fps,
    )
    return {"status": "configured", "camera": health_info.to_dict()}


@app.post("/ingest/frame")
async def ingest_frame(file: UploadFile = File(...)):
    """Push one JPEG frame from mobile camera; repeat continuously."""
    image = _read_image(await file.read())
    # Apply orientation correction once so display + inference agree.
    image = frame_transform(image)
    health_info = _get_camera_manager().ingest_frame(image)
    return {"status": "accepted", "camera": health_info.to_dict()}


@app.post("/ingest/frame/check")
async def ingest_frame_check():
    """Run authorization on the most recent frame in the camera buffer."""
    latest = _get_active_buffer().get_latest()
    if latest is None:
        raise HTTPException(status_code=409, detail="No frames ingested yet — connect camera or POST /ingest/frame")
    _, frame = latest
    return _verify_frame(frame, log_events=True)


@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    """Verify one standalone image against enrolled embeddings."""
    image = _read_image(await file.read())
    return _verify_frame(image, log_events=True)


@app.get("/events")
async def events(limit: int = Query(default=50, ge=1, le=300)):
    items = list(_events_log)[-limit:]
    unauthorized = sum(1 for e in items if e["status"] == "unauthorized")
    return {"count": len(items), "unauthorized_count": unauthorized, "events": items}


# ---------------- Streaming ----------------


@app.get("/stream")
async def stream():
    """Raw MJPEG stream from the active camera."""
    from streamer import mjpeg_from_buffer

    gen = mjpeg_from_buffer(_get_active_buffer())
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stream/detect")
async def stream_detect():
    """Live annotated MJPEG: green=AUTHORIZED, red=UNAUTHORIZED, orange=UNKNOWN."""
    from streamer import mjpeg_from_buffer

    # Orientation is already applied at capture time, so no transform needed here.
    colors = {
        "authorized": (0, 220, 0),
        "unauthorized": (0, 0, 255),
        "unknown": (0, 165, 255),
    }

    def annotate(frame: np.ndarray):
        vis = frame.copy()
        with _detection_lock:
            faces = list(_latest_detections)

        for f in faces:
            x1, y1, x2, y2 = f.get("bbox", [0, 0, 0, 0])
            status = f.get("status", "unknown")
            color = colors.get(status, (255, 255, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            if status == "authorized":
                label = f"AUTHORIZED: {f.get('matched_name', '')}"
            elif status == "unauthorized":
                label = "UNAUTHORIZED"
            else:
                label = "UNKNOWN"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                vis,
                label,
                (x1 + 3, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
        return vis

    gen = mjpeg_from_buffer(_get_active_buffer(), quality=75, transform=annotate)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


# ---- Unified live-stream control (matches carton_counter App 1) ----

def _get_stream() -> MobileCameraStream:
    global _mobile_stream
    if _mobile_stream is None:
        source = os.getenv("VIDEO_SOURCE", "")
        if not source:
            # Default to the mobile IP webcam instead of the laptop (index 0),
            # so the laptop camera never turns on by accident.
            source = os.getenv("MOBILE_IP_CAMERA", "")
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass
        fps = int(os.getenv("STREAM_FPS", "30"))
        _mobile_stream = MobileCameraStream(source=source, fps=fps, transform=_CAMERA_TRANSFORM)
    return _mobile_stream


@app.get("/stream/start")
async def stream_start(source: str = None):
    """Start the unified mobile IP-webcam / device stream (App 1 method).

    Pass `source` (e.g. http://192.168.x.x:8080/video) or set VIDEO_SOURCE env.
    Defaults to MOBILE_IP_CAMERA / VIDEO_SOURCE so the laptop camera is never opened.
    """
    global _mobile_stream
    if _mobile_stream is not None and _mobile_stream.is_active:
        return {"status": "already_running", "source": str(_mobile_stream.source)}
    if source:
        os.environ["VIDEO_SOURCE"] = source
    if not os.getenv("VIDEO_SOURCE", "") and not os.getenv("MOBILE_IP_CAMERA", ""):
        raise HTTPException(
            status_code=400,
            detail="No video source configured. Pass ?source=http://phone-ip:8080/video or set VIDEO_SOURCE / MOBILE_IP_CAMERA env.",
        )
    stream_obj = _get_stream()
    try:
        stream_obj.start()
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"status": "started", "source": str(stream_obj.source)}


@app.get("/stream/stop")
async def stream_stop():
    global _mobile_stream
    if _mobile_stream is not None:
        _mobile_stream.stop()
        _mobile_stream = None
    _get_camera_manager().stop()
    return {"status": "stopped"}


@app.get("/usb/cameras")
async def list_usb_cameras():
    """List available USB cameras without opening them (never powers on laptop webcam)."""
    return {"cameras": list_system_cameras()}


@app.post("/usb/start")
async def usb_start(
    device_index: int = Query(default=0, description="USB device index (0=/dev/video0, 2=/dev/video2)"),
    fps: int = Query(default=20),
):
    """Start capturing from a wired USB webcam."""
    health_info = _get_camera_manager().configure_camera(
        source_type="usb",
        source_uri=str(device_index),
        target_fps=fps,
    )
    return {"status": "started", "camera": health_info.to_dict()}


@app.post("/usb/stop")
async def usb_stop():
    """Stop USB capture."""
    _get_camera_manager().stop()
    return {"status": "stopped"}


@app.post("/camera/connect")
async def camera_connect(
    source_uri: str = Form(..., description="e.g. http://192.168.1.39:8080/video or rtsp://..."),
    fps: int = Form(15, ge=1, le=60),
):
    """Connect directly to an IP Webcam or RTSP stream."""
    health_info = _get_camera_manager().configure_camera(
        source_type="http_mjpeg",
        source_uri=source_uri.strip(),
        target_fps=fps,
    )
    return {"status": "connected", "camera": health_info.to_dict()}


@app.post("/camera/disconnect")
async def camera_disconnect():
    """Disconnect active camera."""
    _get_camera_manager().stop()
    return {"status": "disconnected"}


@app.get("/stream/status")
async def stream_status():
    """Real-time stream health, active source, FPS, counts, and network info."""
    cam_health = _get_camera_manager().get_health()
    buf = _get_active_buffer()
    with _detection_lock:
        dets = list(_latest_detections)
    auth_cnt = sum(1 for d in dets if d.get("status") == "authorized")
    unauth_cnt = sum(1 for d in dets if d.get("status") == "unauthorized")

    return {
        "status": cam_health.status,
        "is_active": buf.is_active,
        "frame_count": buf.frame_count,
        "fps": cam_health.fps,
        "num_faces": len(dets),
        "authorized_count": auth_cnt,
        "unauthorized_count": unauth_cnt,
        "source": cam_health.source_uri,
        "transform": _CAMERA_TRANSFORM,
        "local_ip": _get_local_ip(),
        "port": int(os.getenv("PORT", "8003")),
        "https_port": int(os.getenv("HTTPS_PORT", "8445")),
    }


@app.post("/camera/transform")
async def set_camera_transform(transform: str = Form("none")):
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
    from camera_manager import set_orientation_transform
    set_orientation_transform(_CAMERA_TRANSFORM)
    if _mobile_stream is not None:
        _mobile_stream._transform = _CAMERA_TRANSFORM
    return {"status": "ok", "transform": _CAMERA_TRANSFORM}


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """Stream frames over WebSocket as base64 JPEG (App 1 parity)."""
    import base64

    await websocket.accept()
    if not os.getenv("VIDEO_SOURCE", "") and not os.getenv("MOBILE_IP_CAMERA", ""):
        await websocket.send_json(
            {"type": "error", "detail": "No VIDEO_SOURCE / MOBILE_IP_CAMERA configured for live stream"}
        )
        await websocket.close()
        return
    stream_obj = _get_stream()
    if not stream_obj.is_active:
        try:
            stream_obj.start()
        except RuntimeError as e:
            await websocket.send_json({"type": "error", "detail": str(e)})
            await websocket.close()
            return
    try:
        while True:
            frame = stream_obj.get_frame()
            if frame is None:
                await websocket.send_json({"type": "wait"})
                await websocket.receive_text()  # simple heartbeat / keepalive
                continue
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
            await websocket.send_json({"type": "frame", "data": b64})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/mobile", response_class=HTMLResponse)
async def mobile_camera_page():
    """Mobile HTML5 Camera Push Webpage with HTTPS auto-redirect and native snapshot fallback."""
    local_ip = _get_local_ip()
    port = int(os.getenv("PORT", "8003"))
    https_port = int(os.getenv("HTTPS_PORT", "8445"))

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>Face Auth - Mobile Camera</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 15px; text-align: center; }}
            .header {{ margin-bottom: 12px; }}
            h1 {{ font-size: 1.35rem; color: #38bdf8; margin-bottom: 4px; }}
            p {{ font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px; }}
            
            .https-banner {{ background: #7c2d12; border: 1px solid #ea580c; border-radius: 12px; padding: 12px; margin-bottom: 15px; text-align: left; font-size: 0.85rem; }}
            .https-banner h3 {{ color: #fdba74; font-size: 0.95rem; margin-bottom: 4px; }}
            
            .camera-container {{ position: relative; width: 100%; max-width: 480px; margin: 0 auto 12px; border-radius: 16px; overflow: hidden; background: #1e293b; border: 2px solid #334155; aspect-ratio: 4/3; }}
            video {{ width: 100%; height: 100%; object-fit: cover; }}
            canvas {{ display: none; }}
            .snap-preview {{ max-width: 100%; border-radius: 8px; margin-top: 10px; display: none; }}
            
            .stats-badge {{ position: absolute; top: 10px; left: 10px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(6px); border: 1px solid #38bdf8; border-radius: 20px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600; color: #38bdf8; display: flex; align-items: center; gap: 6px; }}
            .live-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #38bdf8; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.4; transform: scale(0.85); }} }}
            
            .controls {{ display: flex; flex-direction: column; gap: 10px; max-width: 480px; margin: 0 auto; }}
            .btn {{ width: 100%; padding: 13px; border: none; border-radius: 12px; font-size: 0.95rem; font-weight: bold; cursor: pointer; transition: all 0.2s; }}
            .btn-start {{ background: linear-gradient(135deg, #0284c7, #0369a1); color: white; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3); }}
            .btn-snap {{ background: linear-gradient(135deg, #22c55e, #16a34a); color: white; box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3); }}
            .btn-stop {{ background: #ef4444; color: white; display: none; }}
            .btn-switch {{ background: #334155; color: #e2e8f0; }}
            .btn-https {{ background: #ea580c; color: white; margin-top: 8px; }}
            
            .cam-toggle-group {{ display: flex; gap: 8px; max-width: 480px; margin: 0 auto 10px; }}
            .cam-toggle-btn {{ flex: 1; padding: 10px 12px; border-radius: 10px; background: #1e293b; color: #94a3b8; border: 1.5px solid #334155; font-size: 0.85rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px; transition: all 0.2s; }}
            .cam-toggle-btn.active {{ background: #0369a1; color: #fff; border-color: #38bdf8; box-shadow: 0 0 12px rgba(56, 189, 248, 0.3); }}

            .rot-group {{ display: flex; gap: 6px; max-width: 480px; margin: 0 auto 10px; }}
            .rot-btn {{ flex: 1; padding: 8px; border-radius: 8px; background: #1e293b; color: #94a3b8; border: 1px solid #334155; font-size: 0.78rem; cursor: pointer; }}
            .rot-btn.active {{ background: #0284c7; color: white; border-color: #38bdf8; }}

            .camera-container {{ position: relative; width: 100%; max-width: 480px; margin: 0 auto 12px; border-radius: 16px; overflow: hidden; background: #1e293b; border: 2px solid #334155; aspect-ratio: 4/3; }}
            video {{ width: 100%; height: 100%; object-fit: cover; }}
            video.mirror {{ transform: scaleX(-1); }}
            canvas {{ display: none; }}
            .snap-preview {{ max-width: 100%; border-radius: 8px; margin-top: 10px; display: none; }}

            .stats-badge {{ position: absolute; top: 10px; left: 10px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(6px); border: 1px solid #38bdf8; border-radius: 20px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600; color: #38bdf8; display: flex; align-items: center; gap: 6px; }}
            .live-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #38bdf8; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.4; transform: scale(0.85); }} }}

            .controls {{ display: flex; flex-direction: column; gap: 10px; max-width: 480px; margin: 0 auto; }}
            .snap-row {{ display: flex; gap: 8px; }}
            .btn {{ width: 100%; padding: 13px; border: none; border-radius: 12px; font-size: 0.95rem; font-weight: bold; cursor: pointer; transition: all 0.2s; }}
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
            <h1>🔐 Face Auth Mobile Camera</h1>
            <p>Live Stream & Facial Recognition</p>
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
                <span class="status-label">Frames Sent:</span>
                <span class="status-value" id="framesSent">0</span>
            </div>
            <div class="status-row">
                <span class="status-label">Stream Speed:</span>
                <span class="status-value" id="fpsRate">0 fps</span>
            </div>
            <div class="status-row">
                <span class="status-label">Server Target:</span>
                <span class="status-value" style="color: #38bdf8; font-size: 0.78rem;">http://{local_ip}:{port}/ingest/frame</span>
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
            let currentOrientation = 'none';

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

                // Apply mirror CSS on front camera for natural selfie view
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
                currentOrientation = mode;
                document.querySelectorAll('.rot-btn').forEach(b => b.classList.remove('active'));
                const btn = document.getElementById('rot-' + (mode === 'rotate_90_cw' ? 'cw' : mode === 'rotate_90_ccw' ? 'ccw' : mode === 'rotate_180' ? '180' : mode === 'flip_h' ? 'fliph' : 'none'));
                if (btn) btn.classList.add('active');

                const fd = new FormData();
                fd.append('transform', mode);
                try {{
                    await fetch('/camera/transform', {{ method: 'POST', body: fd }});
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
                    document.getElementById('streamStatus').style.color = '#4ade80';

                    if (streamInterval) clearInterval(streamInterval);
                    streamInterval = setInterval(sendFrame, 50);
                }} catch (err) {{
                    document.getElementById('permBanner').style.display = 'block';
                    document.getElementById('streamStatus').textContent = 'Permission Denied / Camera Error';
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
                            let now = Date.now();
                            let elapsed = (now - lastFrameTime) / 1000;
                            if (elapsed > 0) {{
                                document.getElementById('fpsRate').textContent = (1 / elapsed).toFixed(1) + ' fps';
                            }}
                            lastFrameTime = now;
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
                        document.getElementById('streamStatus').textContent = 'Snapshot Verified ✅';
                        document.getElementById('streamStatus').style.color = '#4ade80';

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
    port = int(os.getenv("PORT", "8003"))
    https_port = int(os.getenv("HTTPS_PORT", "8445"))
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face Authorization — Live AI Monitor</title>
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
            .val-auth {{ color: #4ade80; }}
            .val-unauth {{ color: #f87171; }}

            .connection-bar {{
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 14px;
                padding: 14px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.9rem;
            }}
            .btn-action {{
                padding: 8px 16px;
                border-radius: 8px;
                border: none;
                background: #0284c7;
                color: white;
                font-weight: 600;
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <div class="topbar">
            <div class="topbar-title">
                <span>🔐</span> Face Authorization Live Monitor
            </div>
            <div class="status-indicator">
                <span class="status-dot" id="statusDot"></span>
                <span id="statusText">Checking...</span>
            </div>
        </div>

        <div class="main-content">
            <div class="video-container">
                <img id="liveStream" src="/stream/detect" alt="Live Annotated Stream">
                <div class="waiting-overlay" id="waitingOverlay">
                    <div class="spinner"></div>
                    <p style="font-size: 1.1rem; color: #94a3b8;">Connecting to camera feed...</p>
                    <p style="font-size: 0.85rem; color: #64748b;">Open mobile camera on phone or start USB device.</p>
                </div>
            </div>

            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-value val-total" id="kpiFaces">0</div>
                    <div class="kpi-label">Faces Detected</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value val-auth" id="kpiAuth">0</div>
                    <div class="kpi-label">Authorized Persons</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value val-unauth" id="kpiUnauth">0</div>
                    <div class="kpi-label">Unauthorized / Unknown</div>
                </div>
            </div>

            <div class="connection-bar">
                <div>
                    <span>📱 <b>Connect Mobile Camera (HTTPS Secure):</b></span><br>
                    <a href="https://{local_ip}:{https_port}/mobile" target="_blank" style="font-size: 1.05rem; color: #38bdf8;">
                        https://{local_ip}:{https_port}/mobile
                    </a>
                </div>
                <div>
                    <a href="http://{local_ip}:8501" target="_blank" style="color: #60a5fa; text-decoration: none; font-weight: 600;">
                        📊 Open Streamlit Admin Dashboard &rarr;
                    </a>
                </div>
            </div>
        </div>

        <script>
            let waitingOverlay = document.getElementById('waitingOverlay');
            let statusDot = document.getElementById('statusDot');
            let statusText = document.getElementById('statusText');

            async function pollStatus() {{
                try {{
                    let res = await fetch('/stream/status');
                    let data = await res.json();

                    if (data.is_active || data.status === 'connected') {{
                        waitingOverlay.classList.add('hidden');
                        statusDot.classList.add('live');
                        statusText.textContent = 'Live (' + (data.frame_count || 0) + ' frames)';
                    }} else {{
                        waitingOverlay.classList.remove('hidden');
                        statusDot.classList.remove('live');
                        statusText.textContent = 'Standby / Connecting...';
                    }}

                    document.getElementById('kpiFaces').textContent = data.num_faces || 0;
                    document.getElementById('kpiAuth').textContent = data.authorized_count || 0;
                    document.getElementById('kpiUnauth').textContent = data.unauthorized_count || 0;
                }} catch(e) {{}}
            }}

            setInterval(pollStatus, 500);
            pollStatus();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    http_port = int(os.getenv("PORT", "8003"))
    print(f"Starting Face Auth HTTP server on http://0.0.0.0:{http_port}")
    uvicorn.run(app, host="0.0.0.0", port=http_port, log_level="info")
