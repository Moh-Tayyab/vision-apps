"""FastAPI app: Authorized/Unauthorized Person Detection & Security Gate (App 3).

Enterprise-Grade Computer Vision Microservice:
- Passive Anti-Spoofing & Liveness Detection (blocks screen/paper presentation attacks).
- SQLite WAL + Qdrant Vector persistence.
- Multi-target temporal tracking with consensus classification.
- Prometheus /metrics exporter and persistent audit logs.
- API Key / RBAC security on administrative endpoints.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections import deque
from typing import List, Optional

# Limit TensorFlow CPU thread usage BEFORE importing deepface/tensorflow so the
# camera capture + MJPEG encoder are not starved of cores. Inter-op=1 avoids
# context-switch storms that collapse the live stream frame rate.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from anti_spoof import AntiSpoofEngine
from camera_manager import CameraHealth, CameraManager, list_system_cameras
from face_engine import COSINE_THRESHOLD, DETECTOR_BACKEND, FaceEngine
from metrics import metrics
from security import verify_admin_access
from streamer import FrameBuffer, MobileCameraStream, apply_transform
from tracker import FaceTracker, _compute_iou

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("face_auth.main")

app = FastAPI(
    title="Face Authorization",
    description="Enterprise-grade face recognition & access authorization with anti-spoofing and vector search",
    version="2.0.0",
)

DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "face_auth.db")

# Core Engine Singletons
_engine: Optional[FaceEngine] = None
_camera_manager: Optional[CameraManager] = None
_mobile_stream: Optional[MobileCameraStream] = None
_anti_spoof: Optional[AntiSpoofEngine] = None
_tracker: Optional[FaceTracker] = None
_events_log: deque = deque(maxlen=300)

# Orientation correction applied at capture time so display + inference agree.
_CAMERA_TRANSFORM: str = os.getenv("CAMERA_TRANSFORM", "none")
_CAMERA_TRANSFORM_FILE: str = os.getenv("CAMERA_TRANSFORM_FILE", "./data/camera_transform.txt")

# Sensitivity & Detection Parameters
MIN_FACE_SIZE: int = int(os.getenv("MIN_FACE_SIZE", "40"))
DETECTION_CONFIDENCE: float = float(os.getenv("DETECTION_CONFIDENCE", "0.65"))
COSINE_MATCH_THRESHOLD: float = float(os.getenv("COSINE_THRESHOLD", "0.12"))
INFERENCE_MAX_WIDTH: int = int(os.getenv("INFERENCE_MAX_WIDTH", "640"))

# Detection Cache & Async Inference
_detection_lock = threading.Lock()
_latest_detections: list = []
_last_event_timestamps: dict[str, float] = {}  # for event debouncing
_inference_running = True
_inference_thread: Optional[threading.Thread] = None


def _get_engine() -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine(DB_PATH, threshold=COSINE_MATCH_THRESHOLD)
    return _engine


def _get_anti_spoof() -> AntiSpoofEngine:
    global _anti_spoof
    if _anti_spoof is None:
        _anti_spoof = AntiSpoofEngine()
    return _anti_spoof


def _get_tracker() -> FaceTracker:
    global _tracker
    if _tracker is None:
        _tracker = FaceTracker()
    return _tracker


def _reset_tracker() -> None:
    """Clear all tracked persons so identities never carry over between camera sessions."""
    global _tracker
    _tracker = FaceTracker()


def _get_camera_manager() -> CameraManager:
    global _camera_manager
    if _camera_manager is None:
        _camera_manager = CameraManager()
    return _camera_manager


def frame_transform(frame: np.ndarray) -> np.ndarray:
    """Apply the active orientation correction to a frame."""
    return apply_transform(frame, _CAMERA_TRANSFORM)


def _get_active_buffer() -> FrameBuffer:
    """Return the live FrameBuffer being used for display + inference."""
    if _mobile_stream is not None and _mobile_stream.is_active:
        return _mobile_stream.buffer
    return _get_camera_manager().camera.buffer


def _get_local_ip() -> str:
    """LAN IP discovery for mobile instructions."""
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


def _verify_frame(
    image: np.ndarray,
    log_events: bool = True,
    min_face_size: Optional[int] = None,
    confidence_thresh: Optional[float] = None,
    cosine_thresh: Optional[float] = None,
    check_liveness: bool = True,
    use_tracker: bool = True,
) -> dict:
    """Pipeline: Detection -> Passive Liveness -> Vector Search -> Temporal Tracking -> DB Audit."""
    start_t = time.time()
    try:
        from deepface import DeepFace
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"deepface/tensorflow not available in this environment: {e}",
        )

    min_size = min_face_size if min_face_size is not None else MIN_FACE_SIZE
    min_conf = confidence_thresh if confidence_thresh is not None else DETECTION_CONFIDENCE
    eff_cosine = cosine_thresh if cosine_thresh is not None else COSINE_MATCH_THRESHOLD

    h, w = image.shape[:2]
    infer_img = image
    scale_x = 1.0
    scale_y = 1.0
    if w > INFERENCE_MAX_WIDTH:
        infer_w = INFERENCE_MAX_WIDTH
        infer_h = int(h * (INFERENCE_MAX_WIDTH / w))
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

    # Step A: Filter raw face proposals by confidence, size, geometry, and facial landmarks
    candidates = []
    for face in faces:
        facial_area = face.get("facial_area", {})
        confidence = float(face.get("confidence", 0.0))
        if confidence < min_conf:
            continue

        # Strict Facial Landmark Validation:
        # A legitimate human face MUST have left and right eyes detected with realistic separation.
        # This completely rejects shoe racks, striped textures, shirt folds, buttons, and wall reflections!
        left_eye = facial_area.get("left_eye")
        right_eye = facial_area.get("right_eye")
        if not left_eye or not right_eye:
            continue

        eye_dx = abs(left_eye[0] - right_eye[0])
        eye_dy = abs(left_eye[1] - right_eye[1])
        # Eye distance must be at least 10px and mostly horizontal (not vertically aligned)
        if eye_dx < 10 or eye_dy > eye_dx * 1.5:
            continue

        x = int(facial_area.get("x", 0) * scale_x)
        y = int(facial_area.get("y", 0) * scale_y)
        fw = int(facial_area.get("w", 0) * scale_x)
        fh = int(facial_area.get("h", 0) * scale_y)

        if fw < min_size or fh < min_size:
            continue

        # Human face aspect ratio validation (reject bizarre vertical/horizontal light stripes)
        aspect = fw / float(max(1, fh))
        if aspect < 0.55 or aspect > 1.70:
            continue

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + fw)
        y2 = min(h, y + fh)

        candidates.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
            "face": face,
            "fw": fw,
            "fh": fh,
        })

    # Step B: Non-Maximum Suppression (NMS) to eliminate duplicate proposals on the same face
    candidates = sorted(candidates, key=lambda c: c["confidence"], reverse=True)
    kept_candidates = []
    for c in candidates:
        overlap = False
        for k in kept_candidates:
            if _compute_iou(c["bbox"], k["bbox"]) >= 0.35:
                overlap = True
                break
        if not overlap:
            kept_candidates.append(c)

    raw_detections = []
    now = time.time()
    anti_spoof = _get_anti_spoof()
    engine = _get_engine()

    # Below this detected face width embeddings are unreliable and can false-match
    # the WRONG enrolled person. Such faces are reported as "too far" instead of
    # ever guessing an identity. Calibrated against Facenet: bakar@52px -> tayyab.
    FAR_FACE_WIDTH = int(os.getenv("FAR_FACE_WIDTH", "55"))

    for cand in kept_candidates:
        x1, y1, x2, y2 = cand["bbox"]
        fw, fh = cand["fw"], cand["fh"]
        confidence = cand["confidence"]
        face = cand["face"]

        # Face too small => unreliable embedding / wrong-name risk. Never identify.
        if fw < FAR_FACE_WIDTH:
            raw_detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(confidence, 3),
                "liveness_score": 1.0,
                "is_live": True,
                "status": "far",
                "matched_name": None,
                "reason": "face too small - step closer",
                "distance": None,
            })
            continue

        # Extract high-resolution face crop from original un-downscaled image
        # with a 10% safety margin for features
        pad_x = int(fw * 0.10)
        pad_y = int(fh * 0.10)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        orig_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if orig_crop.size > 0:
            face_crop = orig_crop
        else:
            raw_crop = face["face"]
            if np.issubdtype(raw_crop.dtype, np.floating) and raw_crop.max() <= 1.05:
                face_crop = (raw_crop * 255.0).astype(np.uint8)
            else:
                face_crop = raw_crop.astype(np.uint8)

        # Glare / over-exposure rejection: skip pure white light patches
        if face_crop.size > 0:
            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
            if np.mean(gray_crop > 250) > 0.80 or np.mean(gray_crop) > 244:
                continue

        # Step 1: Passive Liveness & Anti-Spoofing Check
        # Only run anti-spoof on legitimate face candidates (confidence >= 0.50)
        liveness_res = anti_spoof.check_liveness(face_crop) if check_liveness else None
        is_live = liveness_res.is_real if liveness_res else True
        liveness_score = liveness_res.liveness_score if liveness_res else 1.0

        entry = {
            "bbox": [x1, y1, x2, y2],
            "confidence": round(confidence, 3),
            "liveness_score": liveness_score,
            "is_live": is_live,
        }

        if not is_live:
            # 2D Presentation attack / screen spoof detected
            entry.update(
                status="spoof",
                matched_name="SPOOF DETECTED",
                reason="presentation_attack",
                distance=None,
            )
        else:
            # Step 2: Vector Search via Database / Qdrant
            match = engine.identify_face(face_crop, threshold=eff_cosine)
            if match is None:
                # No enrolled persons, OR the face crop is too small/degraded to
                # trust an identity (reject ceiling) — never guess a name here.
                entry.update(status="unknown", reason="no match / face too small")
            else:
                entry["matched_name"] = match["name"]
                entry["distance"] = match["distance"]
                entry["status"] = "authorized" if match["authorized"] else "unauthorized"
                entry["vector_engine"] = match.get("engine", "sqlite_numpy")

        raw_detections.append(entry)

    # Step 3: Multi-target Temporal Tracking & Spatial Smoothing
    if use_tracker:
        tracked_results = _get_tracker().update(raw_detections, now=now)
    else:
        # Stateless single-image verification: assign fresh track_ids (1-based) so
        # identity is NEVER carried over from a previous request / camera session.
        tracked_results = []
        for idx, det in enumerate(raw_detections):
            det["track_id"] = idx + 1
            det["age_seconds"] = 0.0
            tracked_results.append(det)

    # Step 4: Persistent Audit Logging
    for entry in tracked_results:
        status_val = entry.get("status", "unknown")
        if log_events and status_val in ("authorized", "unauthorized", "spoof"):
            target_key = f"{entry.get('track_id', 0)}_{entry.get('matched_name', status_val)}"
            last_logged = _last_event_timestamps.get(target_key, 0.0)
            if now - last_logged > 4.0:
                _last_event_timestamps[target_key] = now
                _events_log.append({"timestamp": now, **entry})
                engine.db.log_audit_event(
                    status=status_val,
                    camera_id="live_cam",
                    matched_name=entry.get("matched_name"),
                    confidence=entry.get("confidence", 0.0),
                    distance=entry.get("distance"),
                    liveness_score=entry.get("liveness_score", 1.0),
                    bbox=entry.get("bbox"),
                )

    # Step 5: Prometheus Performance Metrics
    duration_sec = time.time() - start_t
    cam_fps = _get_camera_manager().get_health().fps
    metrics.record_inference(duration_sec, tracked_results, cam_fps)

    return {
        "num_faces": len(tracked_results),
        "faces": tracked_results,
        "any_unauthorized": any(f["status"] == "unauthorized" for f in tracked_results),
        "any_spoof": any(f["status"] == "spoof" for f in tracked_results),
        "inference_latency_ms": round(duration_sec * 1000.0, 1),
    }


def _async_inference_worker():
    """Background worker that continuously runs AI inference on latest camera frame."""
    global _latest_detections

    # Inference is expensive (~100-450ms/frame). Throttle to a target rate so the
    # camera capture loop and MJPEG encoder keep a smooth, higher-FPS stream while
    # detections stay fresh enough for live display + audit.
    min_interval = float(os.getenv("INFERENCE_MIN_INTERVAL", "0.20"))  # ~5 updates/sec

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

        # Cooldown between inference passes
        last_infer_time = getattr(_async_inference_worker, "_last_infer", 0.0)
        elapsed = time.time() - last_infer_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
            continue

        try:
            _async_inference_worker._last_infer = time.time()
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


def _persist_camera_transform(t: str) -> None:
    try:
        p = _CAMERA_TRANSFORM_FILE
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(t)
    except Exception as e:
        print(f"[persist] failed to write transform: {e}")


def _load_camera_transform() -> str:
    try:
        if os.path.exists(_CAMERA_TRANSFORM_FILE):
            with open(_CAMERA_TRANSFORM_FILE) as f:
                v = f.read().strip().lower()
                return v if v else "none"
    except Exception:
        pass
    return "none"


@app.on_event("startup")
async def startup():
    global _inference_thread, _CAMERA_TRANSFORM
    try:
        _persisted = _load_camera_transform()
        if _persisted != _CAMERA_TRANSFORM:
            _CAMERA_TRANSFORM = _persisted
        from camera_manager import set_orientation_transform
        set_orientation_transform(_CAMERA_TRANSFORM)
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

    engine = _get_engine()
    return {
        "library": "deepface",
        "model_name": MODEL_NAME,
        "detector_backend": DETECTOR_BACKEND,
        "cosine_threshold": engine.threshold,
        "min_face_size": MIN_FACE_SIZE,
        "detection_confidence": DETECTION_CONFIDENCE,
        "inference_max_width": INFERENCE_MAX_WIDTH,
        "persons": engine.list_persons(),
    }


# ---------------- Sensitivity & Distance Configuration ----------------


@app.get("/settings/sensitivity")
async def get_sensitivity_settings():
    """Get current face detection sensitivity and distance parameters."""
    engine = _get_engine()
    return {
        "min_face_size": MIN_FACE_SIZE,
        "detection_confidence": round(DETECTION_CONFIDENCE, 3),
        "cosine_match_threshold": round(engine.threshold, 3),
        "inference_max_width": INFERENCE_MAX_WIDTH,
        "detector_backend": DETECTOR_BACKEND,
        "model_name": "Facenet",
    }


@app.post("/settings/sensitivity")
async def update_sensitivity_settings(
    min_face_size: Optional[int] = Form(None, ge=8, le=120),
    detection_confidence: Optional[float] = Form(None, ge=0.05, le=0.95),
    cosine_match_threshold: Optional[float] = Form(None, ge=0.02, le=0.50),
    inference_max_width: Optional[int] = Form(None, ge=320, le=1920),
    preset: Optional[str] = Form(None, description="long_distance | balanced | strict"),
):
    """Adjust detection sensitivity and distance presets dynamically."""
    global MIN_FACE_SIZE, DETECTION_CONFIDENCE, COSINE_MATCH_THRESHOLD, INFERENCE_MAX_WIDTH

    if preset:
        p = preset.lower().strip()
        if p in ("long_distance", "far", "high_sensitivity"):
            MIN_FACE_SIZE = 32
            DETECTION_CONFIDENCE = 0.55
            COSINE_MATCH_THRESHOLD = 0.15
            INFERENCE_MAX_WIDTH = 640
        elif p in ("balanced", "medium", "standard"):
            MIN_FACE_SIZE = 40
            DETECTION_CONFIDENCE = 0.65
            COSINE_MATCH_THRESHOLD = 0.12
            INFERENCE_MAX_WIDTH = 640
        elif p in ("strict", "close", "high_security"):
            MIN_FACE_SIZE = 48
            DETECTION_CONFIDENCE = 0.75
            COSINE_MATCH_THRESHOLD = 0.09
            INFERENCE_MAX_WIDTH = 640
        else:
            raise HTTPException(status_code=422, detail=f"Unknown preset '{preset}'. Choose: long_distance | balanced | strict")

    if min_face_size is not None:
        MIN_FACE_SIZE = int(min_face_size)
    if detection_confidence is not None:
        DETECTION_CONFIDENCE = float(detection_confidence)
    if cosine_match_threshold is not None:
        COSINE_MATCH_THRESHOLD = float(cosine_match_threshold)
    if inference_max_width is not None:
        INFERENCE_MAX_WIDTH = int(inference_max_width)

    _get_engine().threshold = COSINE_MATCH_THRESHOLD

    return {
        "status": "updated",
        "settings": {
            "min_face_size": MIN_FACE_SIZE,
            "detection_confidence": round(DETECTION_CONFIDENCE, 3),
            "cosine_match_threshold": round(COSINE_MATCH_THRESHOLD, 3),
            "inference_max_width": INFERENCE_MAX_WIDTH,
            "detector_backend": DETECTOR_BACKEND,
        },
    }


# ---------------- K-Fold Threshold Calibration ----------------


@app.post("/api/calibrate", dependencies=[Depends(verify_admin_access)])
async def calibrate_threshold(
    k_folds: int = Form(5, ge=2, le=20),
    target_metric: str = Form("eer", description="eer | f1 | youden | accuracy | high_security"),
    apply_calibrated: bool = Form(True, description="Automatically update system threshold if calibrated"),
):
    """Run Stratified K-Fold Cross-Validation Threshold Calibration on currently enrolled persons."""
    from calibration import calibrate_from_db

    try:
        report = calibrate_from_db(DB_PATH, k_folds=k_folds, target_metric=target_metric)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Calibration error: {e}")

    global COSINE_MATCH_THRESHOLD
    if apply_calibrated:
        COSINE_MATCH_THRESHOLD = float(report.recommended_threshold)
        _get_engine().threshold = float(report.recommended_threshold)

    return {
        "status": "calibrated",
        "applied": apply_calibrated,
        "recommended_threshold": report.recommended_threshold,
        "report": report.to_dict(),
    }


@app.get("/api/calibrate/metrics")
async def get_calibration_info():
    """Retrieve current verification threshold and engine parameters."""
    engine = _get_engine()
    return {
        "current_cosine_threshold": round(engine.threshold, 4),
        "detector_backend": DETECTOR_BACKEND,
        "inference_max_width": INFERENCE_MAX_WIDTH,
        "min_face_size": MIN_FACE_SIZE,
        "detection_confidence": DETECTION_CONFIDENCE,
    }


# ---------------- Enrollment & Management (Secured) ----------------


@app.post("/persons/enroll", dependencies=[Depends(verify_admin_access)])
async def enroll(name: str = Form(...), files: List[UploadFile] = File(...)):
    """Save face embeddings for an authorized person (1+ images) with vector persistence."""
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


@app.delete("/persons/{name}", dependencies=[Depends(verify_admin_access)])
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


# ---------------- Production Audit & Metrics Endpoints ----------------


@app.get("/metrics")
async def prometheus_metrics():
    """Expose real-time Prometheus monitoring metrics for scraping."""
    enrolled_count = len(_get_engine().list_persons())
    metrics_text = metrics.generate_prometheus_text(enrolled_persons_count=enrolled_count)
    return Response(content=metrics_text, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/audit/events")
async def audit_events(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None, description="authorized | unauthorized | spoof"),
    name: Optional[str] = Query(default=None, description="Search by person name"),
):
    """Query persistent database audit logs."""
    total, items = _get_engine().db.list_audit_events(limit=limit, offset=offset, status=status, search_name=name)
    return {
        "total": total,
        "count": len(items),
        "offset": offset,
        "limit": limit,
        "events": items,
    }


@app.get("/db/stats")
async def db_stats():
    """Get vector database and persistence engine statistics."""
    return _get_engine().db.get_stats()


# ---------------- Camera Source & Ingestion ----------------


@app.get("/camera/devices")
async def get_camera_devices():
    """List auto-detected USB / V4L2 video devices on the system."""
    return {"devices": list_system_cameras()}


@app.get("/camera/health")
async def get_camera_health():
    """Get active camera health, status, and FPS."""
    return _get_camera_manager().get_health().to_dict()


@app.post("/camera/configure", dependencies=[Depends(verify_admin_access)])
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
    _reset_tracker()
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
    # Detect a NEW mobile session (camera was idle/disconnected) and clear any
    # stale tracked identities from the previous session.
    prev_health = _get_camera_manager().get_health()
    was_idle = prev_health.status in ("standby", "disconnected") or not prev_health.is_connected
    health_info = _get_camera_manager().ingest_frame(image)
    if was_idle:
        _reset_tracker()
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
    return _verify_frame(image, log_events=True, use_tracker=False)


@app.get("/events")
async def events(limit: int = Query(default=50, ge=1, le=300)):
    items = list(_events_log)[-limit:]
    unauthorized = sum(1 for e in items if e["status"] in ("unauthorized", "spoof"))
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
    """Live annotated MJPEG: green=AUTHORIZED, red=UNAUTHORIZED."""
    from streamer import mjpeg_from_buffer

    colors = {
        "authorized": (0, 220, 0),     # Green
        "unauthorized": (0, 0, 255),   # Red
        "spoof": (0, 0, 255),          # Red
        "unknown": (0, 0, 255),        # Red
        "far": (0, 0, 255),            # Red
    }

    def annotate(frame: np.ndarray):
        vis = frame.copy()
        with _detection_lock:
            faces = list(_latest_detections)

        for f in faces:
            x1, y1, x2, y2 = f.get("bbox", [0, 0, 0, 0])
            status = f.get("status", "unknown")
            color = colors.get(status, (0, 0, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            if status == "authorized":
                label = f"AUTHORIZED: {f.get('matched_name', '')}"
            else:
                label = f"UNAUTHORIZED"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                vis,
                label,
                (x1 + 3, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255) if status != "spoof" else (0, 0, 0),
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
    _reset_tracker()
    return {"status": "started", "source": str(stream_obj.source)}


@app.get("/stream/stop")
async def stream_stop():
    global _mobile_stream
    _reset_tracker()
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
    _reset_tracker()
    health_info = _get_camera_manager().configure_camera(
        source_type="usb",
        source_uri=str(device_index),
        target_fps=fps,
    )
    return {"status": "started", "camera": health_info.to_dict()}


@app.post("/usb/stop")
async def usb_stop():
    """Stop USB capture."""
    _reset_tracker()
    _get_camera_manager().stop()
    return {"status": "stopped"}


@app.post("/camera/connect")
async def camera_connect(
    source_uri: str = Form(..., description="e.g. http://192.168.1.39:8080/video or rtsp://..."),
    fps: int = Form(15, ge=1, le=60),
):
    """Connect directly to an IP Webcam or RTSP stream."""
    _reset_tracker()
    health_info = _get_camera_manager().configure_camera(
        source_type="http_mjpeg",
        source_uri=source_uri.strip(),
        target_fps=fps,
    )
    return {"status": "connected", "camera": health_info.to_dict()}


@app.post("/camera/disconnect")
async def camera_disconnect():
    """Disconnect active camera."""
    _reset_tracker()
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
    _persist_camera_transform(_CAMERA_TRANSFORM)
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
                                width: {{ ideal: 1280 }},
                                height: {{ ideal: 720 }}
                            }},
                            audio: false
                        }});
                    }} catch(eExact) {{
                        try {{
                            stream = await navigator.mediaDevices.getUserMedia({{
                                video: {{
                                    facingMode: {{ ideal: facingMode }},
                                    width: {{ ideal: 1280 }},
                                    height: {{ ideal: 720 }}
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
                    streamInterval = setInterval(sendFrame, 33);
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
                const maxW = 640;
                const scale = Math.min(1.0, maxW / video.videoWidth);
                canvas.width = Math.round(video.videoWidth * scale);
                canvas.height = Math.round(video.videoHeight * scale);
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
                }}, 'image/jpeg', 0.65);
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

            <div class="connection-bar" style="background:#0f172a; border-color:#334155; flex-wrap:wrap; gap:10px;">
                <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                    <span>🎯 <b>Distance Sensitivity:</b></span>
                    <button class="btn-action" onclick="setSensitivity('long_distance')" style="background:#0284c7; padding:6px 12px; font-size:0.85rem;">🔭 Long-Distance</button>
                    <button class="btn-action" onclick="setSensitivity('balanced')" style="background:#334155; padding:6px 12px; font-size:0.85rem;">⚖️ Balanced</button>
                    <button class="btn-action" onclick="setSensitivity('strict')" style="background:#334155; padding:6px 12px; font-size:0.85rem;">🔒 Strict</button>
                </div>
                <div id="sensitivityStatus" style="font-size:0.82rem; color:#38bdf8;">
                    Threshold: 0.48 | MinFace: 36px | Conf: 0.55
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

            async function setSensitivity(preset) {{
                let fd = new FormData();
                fd.append('preset', preset);
                try {{
                    await fetch('/settings/sensitivity', {{ method: 'POST', body: fd }});
                    fetchSensitivity();
                }} catch(e) {{}}
            }}

            async function fetchSensitivity() {{
                try {{
                    let res = await fetch('/settings/sensitivity');
                    let d = await res.json();
                    document.getElementById('sensitivityStatus').textContent = 'Match Thresh: ' + d.cosine_match_threshold + ' | MinFace: ' + d.min_face_size + 'px | Conf: ' + d.detection_confidence;
                }} catch(e) {{}}
            }}

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

            fetchSensitivity();
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
