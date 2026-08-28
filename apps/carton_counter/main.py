"""FastAPI app for Carton Counter - pallet counting with YOLO detection."""

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

import os
import sys
import tempfile
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response

from detector import CartonDetector
from counter import CartonCounter, CountResult
from dual_fusion_engine import DualFusionEngine, DualFusionResult, LayerInfo
from streamer import FrameBuffer, MobileCameraStream, mjpeg_from_buffer, websocket_stream, apply_transform
import threading

app = FastAPI(
    title="Carton Counter",
    description="Pallet carton counting system with multi-angle fusion",
    version="1.2.0",
)

_detector: Optional[CartonDetector] = None
_counter: Optional[CartonCounter] = None
_stream: Optional[MobileCameraStream] = None
_camera_transforms: dict[str, str] = {
    "cam1": "none",
    "cam2": "none",
}
_ingest_buffers: dict[str, FrameBuffer] = {
    "cam1": FrameBuffer(max_frames=10),
    "cam2": FrameBuffer(max_frames=10),
}
_camera_detections: dict[str, list] = {
    "cam1": [],
    "cam2": [],
}
_camera_info: dict[str, dict] = {
    "cam1": {"count": 0, "detections": [], "inference_time_ms": 0.0, "timestamp": 0.0, "frames": 0, "source": "mobile"},
    "cam2": {"count": 0, "detections": [], "inference_time_ms": 0.0, "timestamp": 0.0, "frames": 0, "source": "mobile"},
}
_dual_fusion_info: dict = {
    "total_count": 0,
    "layers_count": 0,
    "layers": [],
    "cam1_count": 0,
    "cam2_count": 0,
    "is_dual_active": False,
    "timestamp": 0.0,
}
_last_seen_ts: dict[str, float] = {"cam1": 0.0, "cam2": 0.0}
_detection_lock = threading.Lock()
_worker_running = True


# USB/wired capture workers: cam_id -> threading.Thread
_usb_capture_threads: dict[str, threading.Thread] = {}
_usb_capture_running: dict[str, bool] = {}


def _get_buffer(cam_id: str = "cam1") -> FrameBuffer:
    with _detection_lock:
        if cam_id not in _ingest_buffers:
            _ingest_buffers[cam_id] = FrameBuffer(max_frames=10)
            _camera_detections[cam_id] = []
            _camera_info[cam_id] = {"count": 0, "detections": [], "inference_time_ms": 0.0, "timestamp": 0.0, "frames": 0, "source": "mobile"}
            _last_seen_ts[cam_id] = 0.0
    return _ingest_buffers[cam_id]


def _usb_capture_worker(cam_id: str, device_index: int, fps: int = 30) -> None:
    """Background thread that continuously reads frames from a USB/V4L2 camera device."""
    global _usb_capture_running
    import cv2 as _cv2
    cap = _cv2.VideoCapture(device_index)
    cap.set(_cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(_cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(_cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print(f"[USB] Cannot open device {device_index} for {cam_id}")
        _usb_capture_running[cam_id] = False
        return

    buf = _get_buffer(cam_id)
    print(f"[USB] Camera {cam_id} started from /dev/video{device_index}")
    delay = 1.0 / fps

    while _usb_capture_running.get(cam_id, False):
        ret, frame = cap.read()
        if ret and frame is not None:
            buf.update(frame)
            with _detection_lock:
                if cam_id in _camera_info:
                    _camera_info[cam_id]["source"] = f"usb:/dev/video{device_index}"
        else:
            # Try reconnect once
            cap.release()
            cap = _cv2.VideoCapture(device_index)
        time.sleep(delay)

    cap.release()
    print(f"[USB] Camera {cam_id} capture stopped.")


def start_usb_camera(cam_id: str, device_index: int, fps: int = 30) -> dict:
    """Start a USB webcam capture thread for the given cam_id."""
    global _usb_capture_running, _usb_capture_threads
    # Stop existing thread for this cam if running
    stop_usb_camera(cam_id)
    _usb_capture_running[cam_id] = True
    t = threading.Thread(target=_usb_capture_worker, args=(cam_id, device_index, fps), daemon=True)
    _usb_capture_threads[cam_id] = t
    t.start()
    return {"started": True, "cam_id": cam_id, "device": f"/dev/video{device_index}"}


def stop_usb_camera(cam_id: str) -> None:
    """Stop USB capture thread for a given cam_id."""
    global _usb_capture_running
    if cam_id in _usb_capture_running:
        _usb_capture_running[cam_id] = False
        t = _usb_capture_threads.pop(cam_id, None)
        if t is not None:
            t.join(timeout=2.0)


def _async_detection_worker():
    """Continuous background worker that runs AI inference on fresh video frames from all cameras,
    and fuses dual-camera feeds into a live layer-wise multiplied total count."""
    global _camera_detections, _camera_info, _last_seen_ts, _dual_fusion_info
    while _worker_running:
        try:
            processed_any = False
            for cam_id, buf in list(_ingest_buffers.items()):
                if not buf.is_active:
                    continue
                latest = buf.get_latest()
                if latest is None:
                    continue
                ts, frame = latest
                if ts != _last_seen_ts.get(cam_id, 0.0):
                    _last_seen_ts[cam_id] = ts
                    processed_any = True
                    conf = float(os.getenv("CONF_THRESHOLD", "0.36"))
                    result = _get_detector().detect(frame, confidence=conf)
                    with _detection_lock:
                        prev_source = _camera_info.get(cam_id, {}).get("source", "mobile")
                        _camera_detections[cam_id] = result.detections
                        _camera_info[cam_id] = {
                            "count": len(result.detections),
                            "detections": [d.to_dict() for d in result.detections],
                            "inference_time_ms": result.inference_time_ms,
                            "timestamp": time.time(),
                            "frames": buf.frame_count,
                            "source": prev_source,
                        }

            # Run live dual-camera layer fusion when at least one camera is active
            with _detection_lock:
                c1_dets = list(_camera_detections.get("cam1", []))
                c2_dets = list(_camera_detections.get("cam2", []))
                buf1 = _ingest_buffers.get("cam1")
                buf2 = _ingest_buffers.get("cam2")
                now = time.time()
                c1_active = bool(buf1 and buf1.is_active and (now - _camera_info.get("cam1", {}).get("timestamp", 0) < 6))
                c2_active = bool(buf2 and buf2.is_active and (now - _camera_info.get("cam2", {}).get("timestamp", 0) < 6))

            if c1_active and c2_active:
                h1 = 480
                h2 = 480
                fc = DualFusionEngine.cluster_layers_from_detections(c1_dets, h1)
                sc = DualFusionEngine.cluster_layers_from_detections(c2_dets, h2)
                layers_info, total_cnt = DualFusionEngine.align_and_multiply_layers(fc, sc, h1, h2)
                with _detection_lock:
                    _dual_fusion_info = {
                        "total_count": total_cnt,
                        "layers_count": len(layers_info),
                        "layers": [l.to_dict() for l in layers_info],
                        "cam1_count": len(c1_dets),
                        "cam2_count": len(c2_dets),
                        "is_dual_active": True,
                        "timestamp": now,
                    }
            elif c1_active and c1_dets:
                fc = DualFusionEngine.cluster_layers_from_detections(c1_dets, 480)
                layers_info, total_cnt = DualFusionEngine.align_and_multiply_layers(fc, [], 480, 480)
                with _detection_lock:
                    _dual_fusion_info = {
                        "total_count": total_cnt,
                        "layers_count": len(layers_info),
                        "layers": [l.to_dict() for l in layers_info],
                        "cam1_count": len(c1_dets),
                        "cam2_count": 0,
                        "is_dual_active": False,
                        "timestamp": now,
                    }
            elif c2_active and c2_dets:
                sc = DualFusionEngine.cluster_layers_from_detections(c2_dets, 480)
                layers_info, total_cnt = DualFusionEngine.align_and_multiply_layers([], sc, 480, 480)
                with _detection_lock:
                    _dual_fusion_info = {
                        "total_count": total_cnt,
                        "layers_count": len(layers_info),
                        "layers": [l.to_dict() for l in layers_info],
                        "cam1_count": 0,
                        "cam2_count": len(c2_dets),
                        "is_dual_active": False,
                        "timestamp": now,
                    }

            if not processed_any:
                time.sleep(0.03)
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


_https_started = False


def _start_https_server():
    global _https_started
    if _https_started:
        return
    _https_started = True

    cert_file = os.path.join(os.path.dirname(__file__), "certs", "cert.pem")
    key_file = os.path.join(os.path.dirname(__file__), "certs", "key.pem")
    https_port = int(os.getenv("HTTPS_PORT", "8443"))

    if not (os.path.exists(cert_file) and os.path.exists(key_file)):
        try:
            os.makedirs(os.path.dirname(cert_file), exist_ok=True)
            import subprocess
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_file, "-out", cert_file,
                "-days", "365", "-nodes", "-subj", "/CN=carton-counter"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    if os.path.exists(cert_file) and os.path.exists(key_file):
        try:
            import uvicorn
            print(f"Starting Carton Counter HTTPS server on https://0.0.0.0:{https_port}")
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
        _generate_demo_images()
    except Exception as e:
        print(f"Warning: detector init failed: {e}")
    # Start async AI worker thread for real-time zero-lag stream
    t = threading.Thread(target=_async_detection_worker, daemon=True)
    t.start()
    print("Async background AI detection worker started.")

    https_t = threading.Thread(target=_start_https_server, daemon=True)
    https_t.start()


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
    with _detection_lock:
        active_cams = {cid: info for cid, info in _camera_info.items() if _ingest_buffers.get(cid, FrameBuffer()).is_active}
    return {
        "status": "healthy",
        "backend": detector.backend,
        "active_cameras": list(active_cams.keys()),
        "total_cameras": len(_ingest_buffers),
    }


@app.get("/usb/cameras")
async def list_usb_cameras():
    """Enumerate available V4L2 cameras WITHOUT opening them (avoids powering on the
    internal laptop webcam). Reads device metadata from /sys/class/video4linux."""
    import glob
    import os

    available = []
    for dev in sorted(glob.glob("/dev/video*")):
        idx = int(dev.replace("/dev/video", ""))
        name = ""
        sysfs_name = f"/sys/class/video4linux/video{idx}/name"
        try:
            with open(sysfs_name, "r") as f:
                name = f.read().strip()
        except OSError:
            name = "Unknown"
        is_internal = any(
            kw in name.lower() for kw in ("integrated", "internal", "webcam", "isight", "camera hub")
        )
        available.append({
            "device_index": idx,
            "path": dev,
            "name": name,
            "internal": is_internal,
        })
    return {"cameras": available, "count": len(available)}


@app.post("/usb/start")
async def usb_start(
    cam_id: str = Query(default="cam1", description="Which camera slot to assign (cam1 or cam2)"),
    device_index: int = Query(default=0, description="USB device index (0=/dev/video0, 2=/dev/video2)"),
    fps: int = Query(default=30),
):
    """Start capturing from a wired USB webcam and feed it into the specified camera slot."""
    result = start_usb_camera(cam_id, device_index, fps)
    return result


@app.post("/usb/stop")
async def usb_stop(cam_id: str = Query(default="cam1")):
    """Stop USB webcam capture for a camera slot."""
    stop_usb_camera(cam_id)
    return {"stopped": True, "cam_id": cam_id}


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


def _draw_detections(image: np.ndarray, detections, label_prefix: str = "") -> np.ndarray:
    vis = image.copy()
    h, w = vis.shape[:2]
    count = len(detections)

    # Layer clustering for visual grouping
    layer_clusters = DualFusionEngine.cluster_layers_from_detections(detections, h)
    layer_colors = [
        (34, 197, 94),   # Green
        (59, 130, 246),  # Blue
        (249, 115, 22),  # Orange
        (168, 85, 247),  # Purple
        (236, 72, 153),  # Pink
        (20, 184, 166),  # Teal
        (234, 179, 8),   # Yellow
    ]

    for l_idx, cluster in enumerate(layer_clusters):
        color = layer_colors[l_idx % len(layer_colors)]
        layer_num = l_idx + 1

        # Draw guideline across frame
        c_y1 = min(d.bbox[1] for d in cluster)
        c_y2 = max(d.bbox[3] for d in cluster)
        mid_y = int((c_y1 + c_y2) / 2.0)
        if 0 <= mid_y < h:
            cv2.line(vis, (0, mid_y), (w, mid_y), color, 1, cv2.LINE_AA)

        for col_idx, det in enumerate(cluster):
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"L{layer_num}-{col_idx+1}: {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y1 = max(0, y1 - th - 6)
            label_y2 = y1 if y1 >= th + 6 else y1 + th + 6
            cv2.rectangle(vis, (x1, label_y1), (x1 + tw + 6, label_y2), color, -1)
            cv2.putText(vis, label, (x1 + 3, label_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay Top Header Banner with Total Carton Count & Layer count
    overlay = vis.copy()
    banner_w = min(360, w - 20)
    cv2.rectangle(overlay, (10, 10), (10 + banner_w, 55), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
    cv2.rectangle(vis, (10, 10), (10 + banner_w, 55), (34, 197, 94), 2)
    layer_tag = f" ({len(layer_clusters)} Layers)" if layer_clusters else ""
    cv2.putText(vis, f"{label_prefix}CARTONS: {count}{layer_tag}", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (74, 222, 128), 2, cv2.LINE_AA)
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


@app.post("/count/dual")
async def count_dual(
    front: UploadFile = File(..., description="Front view camera image (Camera 1)"),
    side: UploadFile = File(..., description="Side view camera image (Camera 2)"),
    confidence: Optional[float] = Query(default=None, description="Confidence threshold override"),
    annotate: bool = Query(default=True, description="Whether to include base64 annotated images"),
):
    """Dual-Camera Layer-Wise Pallet Carton Counting Endpoint.

    Identifies horizontal layers on Front and Side faces, computes N1_k * N2_k per layer,
    and returns the total pallet carton count and layer breakdown.
    """
    front_bytes = await front.read()
    side_bytes = await side.read()
    front_img = _read_image(front_bytes)
    side_img = _read_image(side_bytes)

    counter = _get_counter()
    result = counter.count_dual(
        front_image=front_img,
        side_image=side_img,
        confidence=confidence,
        annotate=annotate,
    )
    return result.to_dict()


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


@app.post("/count/pan")
async def count_pan(
    file: Optional[UploadFile] = File(default=None),
    files: Optional[List[UploadFile]] = File(default=None),
    gap_multiplier: float = Query(default=1.7, ge=0.5, le=5.0, description="Multiplier for median gap in layer clustering"),
    sample_interval_sec: float = Query(default=0.6, ge=0.1, le=5.0, description="Frame sampling interval in seconds"),
    confidence: Optional[float] = Query(default=None, description="Detection confidence threshold override"),
    annotate: bool = Query(default=True, description="Whether to include base64 annotated frames in response"),
):
    """Per-Layer Carton Counting endpoint for vertical pan/tilt pallet videos or image sequences.

    Clusters cartons into physical horizontal layers and sums de-duplicated cartons per layer.
    """
    counter = _get_counter()

    # Case 1: Uploaded multiple image frames
    if files and len(files) > 0:
        frames = []
        for f in files:
            contents = await f.read()
            frames.append(_read_image(contents))
        result = counter.count_pan(
            frames=frames,
            gap_multiplier=gap_multiplier,
            confidence=confidence,
            annotate=annotate,
        )
        return result.to_dict()

    # Case 2: Uploaded video file
    if file is not None:
        contents = await file.read()
        suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(contents)
            tmp.close()
            result = counter.count_pan_video(
                video_path=tmp.name,
                sample_interval_sec=sample_interval_sec,
                gap_multiplier=gap_multiplier,
                confidence=confidence,
                annotate=annotate,
            )
            return result.to_dict()
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    raise HTTPException(
        status_code=400,
        detail="Must provide either a video file ('file') or multiple image frames ('files')",
    )


@app.post("/count/video")
async def count_video(
    file: UploadFile = File(...),
    method: str = Query(default="per_layer_pan", enum=["per_layer_pan", "multi_frame_voting"]),
    sample_frames: int = Query(default=10, ge=1, le=60),
    sample_interval_sec: float = Query(default=0.6, ge=0.1, le=5.0),
    gap_multiplier: float = Query(default=1.7, ge=0.5, le=5.0),
    confidence: Optional[float] = Query(default=None),
    annotate: bool = Query(default=False),
):
    """Process video for carton counting using either per_layer_pan or multi_frame_voting."""
    contents = await file.read()
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(contents)
        tmp.close()

        counter = _get_counter()

        if method == "per_layer_pan":
            result = counter.count_pan_video(
                video_path=tmp.name,
                sample_interval_sec=sample_interval_sec,
                gap_multiplier=gap_multiplier,
                confidence=confidence,
                annotate=annotate,
            )
            return result.to_dict()

        # Legacy multi_frame_voting
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

        result = counter.count_multi_frame(frames)
        return result.to_dict()
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


@app.post("/ingest/frame")
async def ingest_frame(
    file: UploadFile = File(...),
    camera_id: str = Query(default="cam1", description="Camera identifier (e.g. cam1, cam2, front, side)"),
):
    """Accept a frame pushed from any mobile camera client immediately without latency."""
    image = _read_image(await file.read())
    tf = _camera_transforms.get(camera_id, "none")
    image = apply_transform(image, tf)
    buf = _get_buffer(camera_id)
    buf.update(image)

    with _detection_lock:
        info = _camera_info.get(camera_id, {})
        current_count = info.get("count", 0)

    return {
        "status": "accepted",
        "camera_id": camera_id,
        "ingest_frames": buf.frame_count,
        "count": current_count,
        "size": [image.shape[1], image.shape[0]],
    }


@app.post("/camera/transform")
async def set_camera_transform(
    camera_id: str = Query(default="cam1"),
    transform: str = Query(default="none"),
):
    """Live-adjust orientation for a camera slot."""
    global _camera_transforms
    valid = ("none", "flip_h", "flip_v", "rotate_90_cw", "rotate_90_ccw", "rotate_180", "rotate_90", "rotate_270")
    t_clean = transform.lower().strip()
    if t_clean not in valid:
        raise HTTPException(status_code=422, detail=f"Invalid transform '{transform}'. Must be one of: {valid}")
    if t_clean in ("rotate_90", "90"):
        t_clean = "rotate_90_cw"
    elif t_clean in ("rotate_270", "270"):
        t_clean = "rotate_90_ccw"
    _camera_transforms[camera_id] = t_clean
    return {"status": "ok", "camera_id": camera_id, "transform": t_clean}


@app.get("/stream")
async def stream(cam: str = Query(default="cam1")):
    """MJPEG live view for a specific camera (defaults to cam1)."""
    buf = _get_buffer(cam)
    gen = mjpeg_from_buffer(buf, fps_limit=30.0)
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/detect")
async def stream_detect(
    cam: str = Query(default="cam1"),
    confidence: Optional[float] = Query(default=None),
):
    """High-speed 30 FPS MJPEG stream with async bounding boxes overlay for a specific camera."""
    def annotate(frame: np.ndarray) -> np.ndarray:
        with _detection_lock:
            current_dets = list(_camera_detections.get(cam, []))
        label_prefix = f"CAM {cam.upper().replace('CAM', '')} " if len(cam) <= 6 else f"{cam.upper()}: "
        return _draw_detections(frame, current_dets, label_prefix=label_prefix)

    buf = _get_buffer(cam)
    gen = mjpeg_from_buffer(buf, quality=75, transform=annotate, fps_limit=30.0)
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame",
    )



@app.get("/stream/status")
async def stream_status():
    stream_obj = _stream
    with _detection_lock:
        cameras_data = {}
        total_live_count = 0
        now = time.time()
        for cid, info in _camera_info.items():
            buf = _ingest_buffers.get(cid)
            active = buf.is_active if buf else False
            ts = info.get("timestamp", 0.0)
            fresh = active and (now - ts < 6)
            cnt = info.get("count", 0) if fresh else 0
            cameras_data[cid] = {
                "active": fresh,
                "frames": buf.frame_count if buf else 0,
                "count": cnt,
                "timestamp": ts,
                "source": info.get("source", "mobile"),
            }
            if fresh:
                total_live_count += cnt

        cam1_info = _camera_info.get("cam1", {"count": 0, "timestamp": 0.0, "detections": []})
        cam1_buf = _ingest_buffers.get("cam1")
        dual_data = dict(_dual_fusion_info)

    # Use layer-multiplied count if dual cameras are active, otherwise visible sum
    final_total = dual_data.get("total_count", total_live_count) if dual_data.get("is_dual_active") else total_live_count

    return {
        "cameras": cameras_data,
        "total_count": final_total,
        "dual_fusion": dual_data,
        # Backwards compatibility fields for cam1
        "ingest_active": cam1_buf.is_active if cam1_buf else False,
        "ingest_frames": cam1_buf.frame_count if cam1_buf else 0,
        "stream_active": stream_obj.is_active if stream_obj else False,
        "latest_detection": cam1_info,
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
async def mobile_camera_page(cam: str = Query(default="cam1")):
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
            .header {{ margin-bottom: 10px; }}
            h1 {{ font-size: 1.35rem; color: #60a5fa; margin-bottom: 4px; }}
            p {{ font-size: 0.85rem; color: #94a3b8; }}

            /* ── Camera Selector Tabs ── */
            .cam-selector {{
                display: flex;
                gap: 8px;
                max-width: 460px;
                margin: 0 auto 12px;
                background: #1e293b;
                padding: 4px;
                border-radius: 12px;
            }}
            .cam-tab {{
                flex: 1;
                padding: 10px;
                border-radius: 8px;
                font-size: 0.85rem;
                font-weight: bold;
                color: #94a3b8;
                text-decoration: none;
                background: transparent;
                border: none;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .cam-tab.active {{
                background: #3b82f6;
                color: white;
                box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4);
            }}

            .https-banner {{ background: #7c2d12; border: 1px solid #ea580c; border-radius: 12px; padding: 12px; margin-bottom: 15px; text-align: left; font-size: 0.85rem; }}
            .https-banner h3 {{ color: #fdba74; font-size: 0.95rem; margin-bottom: 4px; }}
            .camera-container {{ position: relative; width: 100%; max-width: 460px; margin: 0 auto 12px; border-radius: 16px; overflow: hidden; background: #1e293b; border: 2px solid #334155; aspect-ratio: 4/3; }}
            video {{ width: 100%; height: 100%; object-fit: cover; }}
            canvas {{ display: none; }}
            .stats-badge {{ position: absolute; top: 10px; left: 10px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(6px); border: 1px solid #4ade80; border-radius: 20px; padding: 5px 12px; font-size: 0.8rem; font-weight: 600; color: #4ade80; display: flex; align-items: center; gap: 6px; }}
            .live-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #4ade80; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.4; transform: scale(0.85); }} }}
            .controls {{ display: flex; flex-direction: column; gap: 10px; max-width: 460px; margin: 0 auto; }}
            .snap-row {{ display: flex; gap: 8px; }}
            .btn {{ width: 100%; padding: 14px; border: none; border-radius: 12px; font-size: 1rem; font-weight: bold; cursor: pointer; transition: all 0.2s; }}
            .btn-start {{ background: linear-gradient(135deg, #22c55e, #16a34a); color: white; box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3); }}
            .btn-snap {{ background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); font-size: 0.88rem; padding: 11px; }}
            .btn-snap-user {{ background: linear-gradient(135deg, #8b5cf6, #7c3aed); color: white; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3); font-size: 0.88rem; padding: 11px; }}
            .btn-stop {{ background: #ef4444; color: white; display: none; }}
            .btn-https {{ background: #ea580c; color: white; margin-top: 6px; }}
            
            .cam-toggle-group {{ display: flex; gap: 8px; max-width: 460px; margin: 0 auto 10px; }}
            .cam-toggle-btn {{ flex: 1; padding: 10px 12px; border-radius: 10px; background: #1e293b; color: #94a3b8; border: 1.5px solid #334155; font-size: 0.85rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px; transition: all 0.2s; }}
            .cam-toggle-btn.active {{ background: #0369a1; color: #fff; border-color: #38bdf8; box-shadow: 0 0 12px rgba(56, 189, 248, 0.3); }}

            video {{ width: 100%; height: 100%; object-fit: cover; }}
            video.mirror {{ transform: scaleX(-1); }}

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
            <p>Select Camera Slot & Start Streaming</p>
        </div>

        <!-- Camera Slot Selector -->
        <div class="cam-selector">
            <button class="cam-tab {'active' if cam=='cam1' else ''}" onclick="selectCamera('cam1')">📷 Camera 1 (Front Face)</button>
            <button class="cam-tab {'active' if cam=='cam2' else ''}" onclick="selectCamera('cam2')">📷 Camera 2 (Side Face)</button>
        </div>

        <div id="httpsBanner" class="https-banner" style="display: none;">
            <h3>🔒 Live Video requires HTTPS</h3>
            <p>Mobile browsers only allow live video streaming over HTTPS. Tap below to switch to HTTPS (if prompted, tap <i>Advanced &rarr; Proceed</i>):</p>
            <button class="btn btn-https" onclick="switchToHttps()">🔒 Switch to HTTPS Stream</button>
        </div>

        <div id="permBanner" class="https-banner" style="display: none; background: #450a0a; border-color: #ef4444;">
            <h3 style="color: #fca5a5;">⚠️ Camera Permission Denied</h3>
            <p style="margin-bottom: 8px;">The browser did not grant camera access. You have 2 options to proceed:</p>
            <ol style="text-align: left; padding-left: 20px; font-size: 0.8rem; line-height: 1.5; color: #fecaca;">
                <li>Tap the <b>🔒 Lock / Tune icon</b> next to the URL &rarr; <b>Permissions</b> &rarr; Set <b>Camera: Allow</b>, then refresh.</li>
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
        <div class="rot-group" style="display: flex; gap: 6px; max-width: 460px; margin: 0 auto 10px;">
            <button class="rot-btn active" id="rot-none" onclick="setOrientation('none')" style="flex:1; padding:8px; border-radius:8px; background:#1e293b; color:#94a3b8; border:1px solid #334155; font-size:0.78rem; cursor:pointer;">Normal</button>
            <button class="rot-btn" id="rot-cw" onclick="setOrientation('rotate_90_cw')" style="flex:1; padding:8px; border-radius:8px; background:#1e293b; color:#94a3b8; border:1px solid #334155; font-size:0.78rem; cursor:pointer;">🔄 90° CW</button>
            <button class="rot-btn" id="rot-ccw" onclick="setOrientation('rotate_90_ccw')" style="flex:1; padding:8px; border-radius:8px; background:#1e293b; color:#94a3b8; border:1px solid #334155; font-size:0.78rem; cursor:pointer;">🔄 90° CCW</button>
            <button class="rot-btn" id="rot-180" onclick="setOrientation('rotate_180')" style="flex:1; padding:8px; border-radius:8px; background:#1e293b; color:#94a3b8; border:1px solid #334155; font-size:0.78rem; cursor:pointer;">🔄 180°</button>
            <button class="rot-btn" id="rot-fliph" onclick="setOrientation('flip_h')" style="flex:1; padding:8px; border-radius:8px; background:#1e293b; color:#94a3b8; border:1px solid #334155; font-size:0.78rem; cursor:pointer;">🪞 Flip</button>
        </div>

        <div class="camera-container">
            <video id="video" autoplay playsinline muted></video>
            <img id="snapPreview" class="snap-preview" alt="Captured Frame">
            <canvas id="canvas"></canvas>
            <div class="stats-badge" id="liveBadge" style="display: none;">
                <span class="live-dot"></span> <span id="badgeText">STREAMING: {cam.upper()}</span>
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
                <span class="status-label">Active Camera Lens:</span>
                <span class="status-value" id="activeCamText" style="color: #38bdf8;">Back Camera (Main)</span>
            </div>
            <div class="status-row">
                <span class="status-label">Active Camera Slot:</span>
                <span class="status-value" id="currentSlotText" style="color: #60a5fa;">{cam.upper()}</span>
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
        </div>

        <div class="links">
            <a class="link-btn" href="/" target="_blank">💻 View Dual-Camera Dashboard on Laptop</a>
        </div>

        <script>
            let activeCam = '{cam}';
            let video = document.getElementById('video');
            let canvas = document.getElementById('canvas');
            let snapPreview = document.getElementById('snapPreview');
            let stream = null;
            let streamInterval = null;
            let facingMode = 'environment';
            let frameCount = 0;
            let lastFrameTime = Date.now();

            function selectCamera(slot) {{
                const url = new URL(window.location.href);
                url.searchParams.set('cam', slot);
                window.location.href = url.toString();
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
                document.querySelectorAll('.rot-btn').forEach(b => {{
                    b.style.background = '#1e293b';
                    b.style.color = '#94a3b8';
                    b.style.borderColor = '#334155';
                }});
                const btn = document.getElementById('rot-' + (mode === 'rotate_90_cw' ? 'cw' : mode === 'rotate_90_ccw' ? 'ccw' : mode === 'rotate_180' ? '180' : mode === 'flip_h' ? 'fliph' : 'none'));
                if (btn) {{
                    btn.style.background = '#3b82f6';
                    btn.style.color = 'white';
                    btn.style.borderColor = '#60a5fa';
                }}
                try {{
                    await fetch('/camera/transform?camera_id=' + encodeURIComponent(activeCam) + '&transform=' + encodeURIComponent(mode), {{ method: 'POST' }});
                }} catch(e) {{}}
            }}

            window.onload = function() {{
                if (window.location.protocol === 'http:' && (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)) {{
                    document.getElementById('httpsBanner').style.display = 'block';
                }}
            }};

            function switchToHttps() {{
                window.location.href = 'https://' + window.location.hostname + ':' + (window.location.port || '8001') + '/mobile?cam=' + activeCam;
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
                            stream = await navigator.mediaDevices.getUserMedia({{
                                video: true,
                                audio: false
                            }});
                        }}
                    }}

                    video.style.display = 'block';
                    snapPreview.style.display = 'none';
                    video.srcObject = stream;
                    await video.play();

                    document.getElementById('startBtn').style.display = 'none';
                    document.getElementById('stopBtn').style.display = 'block';
                    document.getElementById('liveBadge').style.display = 'flex';
                    document.getElementById('streamStatus').textContent = 'Streaming Live (' + activeCam.toUpperCase() + ')';
                    document.getElementById('streamStatus').style.color = '#4ade80';

                    if (streamInterval) clearInterval(streamInterval);
                    streamInterval = setInterval(sendFrame, 50);
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
                    fetch('/ingest/frame?camera_id=' + encodeURIComponent(activeCam), {{ method: 'POST', body: fd }})
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
                fetch('/ingest/frame?camera_id=' + encodeURIComponent(activeCam), {{ method: 'POST', body: fd }})
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
        <title>Carton Counter — Dual-Camera Live Monitor</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: #0b0f19; color: #f8fafc; display: flex; flex-direction: column; align-items: center; min-height: 100vh; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}

            /* ── TOP BAR ── */
            .topbar {{
                width: 100%;
                background: #0f172a;
                border-bottom: 1px solid #1e293b;
                padding: 14px 24px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .topbar-title {{ color: #60a5fa; font-size: 1.2rem; font-weight: bold; display: flex; align-items: center; gap: 8px; }}
            .status-dot {{ width: 10px; height: 10px; border-radius: 50%; background: #ef4444; display: inline-block; }}
            .status-dot.live {{ background: #22c55e; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:0.3 }} }}

            /* ── MAIN CONTENT ── */
            .main-content {{
                width: 100%;
                max-width: 1200px;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}

            /* ── TOTAL SUMMARY CARD ── */
            .total-banner {{
                background: linear-gradient(135deg, #1e293b, #0f172a);
                border: 2px solid #3b82f6;
                border-radius: 16px;
                padding: 20px 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 15px;
            }}
            .total-title {{ font-size: 1.1rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }}
            .total-subtitle {{ color: #64748b; font-size: 0.85rem; margin-top: 4px; }}
            .total-count-num {{ font-size: 3.8rem; font-weight: 900; color: #4ade80; line-height: 1; }}

            /* ── DUAL CAMERA GRID ── */
            .cameras-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
                gap: 20px;
                width: 100%;
            }}
            .camera-card {{
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 4px 20px rgba(0,0,0,0.4);
                display: flex;
                flex-direction: column;
            }}
            .card-header {{
                padding: 12px 18px;
                background: #0f172a;
                border-bottom: 1px solid #334155;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .card-title {{ font-weight: bold; font-size: 0.95rem; display: flex; align-items: center; gap: 8px; }}
            .video-wrapper {{
                position: relative;
                width: 100%;
                aspect-ratio: 4/3;
                background: #000;
            }}
            .stream-img {{
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
                background: #0f172a;
                color: #64748b;
                font-size: 0.95rem;
                gap: 12px;
                padding: 20px;
                text-align: center;
            }}
            .waiting-overlay.hidden {{ display: none; }}
            .spinner {{ width: 36px; height: 36px; border: 3px solid #334155; border-top-color: #38bdf8; border-radius: 50%; animation: spin 1s linear infinite; }}
            @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

            .card-footer {{
                padding: 14px 18px;
                background: #111827;
                border-top: 1px solid #334155;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .cam-count {{ font-size: 1.8rem; font-weight: 900; color: #4ade80; }}

            /* ── LAYER BREAKDOWN (CLEAN DETAILS CARD) ── */
            .layer-details-card {{
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 16px;
                padding: 18px 22px;
                display: none;
            }}
            .layer-details-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            }}
            .layer-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.88rem;
                text-align: left;
            }}
            .layer-table th {{
                padding: 8px 12px;
                border-bottom: 2px solid #334155;
                color: #94a3b8;
                font-weight: 600;
            }}
            .layer-table td {{
                padding: 10px 12px;
                border-bottom: 1px solid #2d3748;
                color: #e2e8f0;
            }}

            /* ── CONNECTION PANEL ── */
            .links-box {{
                background: #1e293b;
                border-radius: 14px;
                padding: 16px 20px;
                display: flex;
                flex-direction: column;
                gap: 10px;
                border: 1px solid #334155;
            }}
            .link-row {{ display: flex; justify-content: space-between; align-items: center; font-size: 0.9rem; flex-wrap: wrap; gap: 6px; }}
            .link-val {{ color: #38bdf8; font-weight: bold; text-decoration: none; word-break: break-all; }}
        </style>
    </head>
    <body>

        <!-- TOP BAR -->
        <div class="topbar">
            <div class="topbar-title">📦 Carton Counter — Multi-Camera Live Monitor</div>
            <div style="font-size: 0.9rem; color: #94a3b8;">
                <span class="status-dot" id="globalDot"></span> <span id="globalStatusText">Checking cameras...</span>
            </div>
        </div>

        <div class="main-content">

            <!-- TOTAL SUMMARY CARD -->
            <div class="total-banner">
                <div>
                    <div class="total-title">Total Pallet Cartons</div>
                    <div class="total-subtitle" id="totalSubtitle">Combined count across active camera feeds</div>
                </div>
                <div class="total-count-num" id="totalCountNum">—</div>
            </div>

            <!-- DUAL CAMERA GRID -->
            <div class="cameras-grid">

                <!-- CAMERA 1 (FRONT FACE) -->
                <div class="camera-card">
                    <div class="card-header">
                        <div class="card-title">
                            <span class="status-dot" id="dot1"></span> 📷 Camera 1 (Front Face)
                        </div>
                        <span id="badge1" style="font-size: 0.8rem; color: #64748b;">Waiting...</span>
                    </div>
                    <div class="video-wrapper">
                        <img class="stream-img" id="stream1" src="/stream/detect?cam=cam1" alt="Camera 1 Stream">
                        <div class="waiting-overlay" id="overlay1">
                            <div class="spinner"></div>
                            <span>Camera 1 Connect Hone Ka Intezaar Hai…</span>
                            <small style="color:#64748b">Mobile 1 par <b>Camera 1</b> link kholein</small>
                        </div>
                    </div>
                    <div class="card-footer">
                        <span style="font-size: 0.85rem; color: #94a3b8;">Front Cartons:</span>
                        <span class="cam-count" id="count1">—</span>
                    </div>
                </div>

                <!-- CAMERA 2 (SIDE FACE) -->
                <div class="camera-card">
                    <div class="card-header">
                        <div class="card-title">
                            <span class="status-dot" id="dot2"></span> 📷 Camera 2 (Side Face)
                        </div>
                        <span id="badge2" style="font-size: 0.8rem; color: #64748b;">Waiting...</span>
                    </div>
                    <div class="video-wrapper">
                        <img class="stream-img" id="stream2" src="/stream/detect?cam=cam2" alt="Camera 2 Stream">
                        <div class="waiting-overlay" id="overlay2">
                            <div class="spinner"></div>
                            <span>Camera 2 Connect Hone Ka Intezaar Hai…</span>
                            <small style="color:#64748b">Mobile 2 par <b>Camera 2</b> link kholein</small>
                        </div>
                    </div>
                    <div class="card-footer">
                        <span style="font-size: 0.85rem; color: #94a3b8;">Side Cartons:</span>
                        <span class="cam-count" id="count2">—</span>
                    </div>
                </div>

            </div>

            <!-- LAYER-WISE BREAKDOWN CARD (Clean & Compact) -->
            <div class="layer-details-card" id="layerDetailsCard">
                <div class="layer-details-header">
                    <h3 style="font-size: 1rem; color: #60a5fa; display: flex; align-items: center; gap: 8px;">
                        📊 Live Pallet Layer Breakdown (Front × Side)
                    </h3>
                    <span id="layerSummaryBadge" style="font-size: 0.8rem; background: #0f172a; border: 1px solid #3b82f6; color: #38bdf8; padding: 3px 10px; border-radius: 12px;">0 Layers</span>
                </div>
                <table class="layer-table">
                    <thead>
                        <tr>
                            <th>Layer</th>
                            <th>Front View (N1)</th>
                            <th>Side View (N2)</th>
                            <th>Layer Formula</th>
                            <th style="text-align: right;">Layer Total</th>
                        </tr>
                    </thead>
                    <tbody id="layerTableBody">
                    </tbody>
                </table>
            </div>

            <!-- CONNECTION PANEL: Mobile WiFi + USB Wired -->
            <div class="links-box">
                <div style="display: flex; gap: 8px; margin-bottom: 12px;">
                    <button id="tabMobile" onclick="showTab('mobile')"
                        style="flex:1; padding:10px; border-radius:8px; border:none; cursor:pointer;
                               background:#3b82f6; color:white; font-weight:bold; font-size:0.9rem;">
                        📱 Wireless (Mobile Wi-Fi)
                    </button>
                    <button id="tabUsb" onclick="showTab('usb')"
                        style="flex:1; padding:10px; border-radius:8px; border:none; cursor:pointer;
                               background:#334155; color:#94a3b8; font-weight:bold; font-size:0.9rem;">
                        🔌 Wired (USB Webcam)
                    </button>
                </div>

                <!-- Mobile Tab -->
                <div id="panelMobile">
                    <div class="link-row">
                        <span style="color: #94a3b8;">📱 Mobile 1 (Camera 1 — Front Face):</span>
                        <a class="link-val" href="/mobile?cam=cam1" target="_blank">/mobile?cam=cam1</a>
                    </div>
                    <div class="link-row" style="margin-top:8px;">
                        <span style="color: #94a3b8;">📱 Mobile 2 (Camera 2 — Side Face):</span>
                        <a class="link-val" href="/mobile?cam=cam2" target="_blank">/mobile?cam=cam2</a>
                    </div>
                    <div style="margin-top:10px; font-size:0.8rem; color:#64748b;">
                        ✅ Open link on both mobile phones → tap "Start Live Video Stream"
                    </div>
                </div>

                <!-- USB Wired Tab -->
                <div id="panelUsb" style="display:none;">
                    <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:12px;">
                        Assign USB webcams (connected to the computer) directly to camera slots.
                    </div>

                    <div class="link-row" style="margin-bottom:10px;">
                        <span style="color:#94a3b8;">Available USB Cameras:</span>
                        <button onclick="detectUsbCams()"
                            style="padding:6px 14px; border-radius:6px; border:none; cursor:pointer;
                                   background:#1e40af; color:white; font-size:0.85rem;">🔍 Detect Cameras</button>
                    </div>
                    <div id="usbCamList" style="background:#0f172a; border-radius:8px; padding:10px; font-size:0.82rem;
                        color:#64748b; margin-bottom:12px; min-height:36px;">
                        Click "Detect Cameras"...
                    </div>

                    <div class="link-row" style="margin-bottom:8px; flex-wrap:nowrap; gap:8px;">
                        <label style="color:#94a3b8; font-size:0.85rem; white-space:nowrap;">📷 Camera 1 Device:</label>
                        <select id="usbDev1" style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                            border-radius:6px; padding:5px 8px; font-size:0.82rem;">
                            <option value="0">/dev/video0 (default)</option>
                            <option value="1">/dev/video1</option>
                            <option value="2">/dev/video2</option>
                            <option value="4">/dev/video4</option>
                        </select>
                        <button onclick="startUsbCam('cam1', document.getElementById('usbDev1').value)"
                            style="padding:6px 12px; border-radius:6px; border:none; cursor:pointer;
                                   background:#16a34a; color:white; font-size:0.82rem; white-space:nowrap;">▶ Start</button>
                        <button onclick="stopUsbCam('cam1')"
                            style="padding:6px 12px; border-radius:6px; border:none; cursor:pointer;
                                   background:#7f1d1d; color:#fca5a5; font-size:0.82rem; white-space:nowrap;">■ Stop</button>
                    </div>

                    <div class="link-row" style="flex-wrap:nowrap; gap:8px;">
                        <label style="color:#94a3b8; font-size:0.85rem; white-space:nowrap;">📷 Camera 2 Device:</label>
                        <select id="usbDev2" style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                            border-radius:6px; padding:5px 8px; font-size:0.82rem;">
                            <option value="2">/dev/video2 (default)</option>
                            <option value="0">/dev/video0</option>
                            <option value="1">/dev/video1</option>
                            <option value="4">/dev/video4</option>
                        </select>
                        <button onclick="startUsbCam('cam2', document.getElementById('usbDev2').value)"
                            style="padding:6px 12px; border-radius:6px; border:none; cursor:pointer;
                                   background:#16a34a; color:white; font-size:0.82rem; white-space:nowrap;">▶ Start</button>
                        <button onclick="stopUsbCam('cam2')"
                            style="padding:6px 12px; border-radius:6px; border:none; cursor:pointer;
                                   background:#7f1d1d; color:#fca5a5; font-size:0.82rem; white-space:nowrap;">■ Stop</button>
                    </div>

                    <div id="usbStatus" style="margin-top:10px; font-size:0.82rem; color:#64748b;"></div>
                </div>

                <!-- Swagger Link -->
                <div class="link-row" style="margin-top:12px; padding-top:12px; border-top:1px solid #334155;">
                    <span style="color: #94a3b8;">⚙️ Swagger API Docs:</span>
                    <a class="link-val" href="/docs" target="_blank">/docs</a>
                </div>
            </div>

        </div>

        <script>
            // ── Tab Switcher ──
            function showTab(tab) {{
                const isMobile = tab === 'mobile';
                document.getElementById('panelMobile').style.display = isMobile ? 'block' : 'none';
                document.getElementById('panelUsb').style.display = isMobile ? 'none' : 'block';
                document.getElementById('tabMobile').style.background = isMobile ? '#3b82f6' : '#334155';
                document.getElementById('tabMobile').style.color = isMobile ? 'white' : '#94a3b8';
                document.getElementById('tabUsb').style.background = isMobile ? '#334155' : '#3b82f6';
                document.getElementById('tabUsb').style.color = isMobile ? '#94a3b8' : 'white';
            }}

            // ── USB Camera Helpers ──
            async function detectUsbCams() {{
                const el = document.getElementById('usbCamList');
                el.textContent = 'Detecting...';
                try {{
                    const res = await fetch('/usb/cameras');
                    const data = await res.json();
                    if (data.cameras.length === 0) {{
                        el.textContent = '⚠️ No USB cameras detected. Please check cables.';
                        el.style.color = '#f59e0b';
                    }} else {{
                        el.style.color = '#4ade80';
                        el.innerHTML = data.cameras.map(c => {{
                            const tag = c.internal ? ' 💻 (laptop built-in)' : ' 🔌 (external USB)';
                            return `✅ ${{c.path}} — ${{c.name}}${{tag}}`;
                        }}).join('<br>');
                    }}
                }} catch(e) {{
                    el.textContent = 'Error: ' + e.message;
                    el.style.color = '#ef4444';
                }}
            }}

            async function startUsbCam(camId, deviceIndex) {{
                const st = document.getElementById('usbStatus');
                st.textContent = `${{camId}} start ho raha hai /dev/video${{deviceIndex}} se...`;
                st.style.color = '#60a5fa';
                try {{
                    const res = await fetch(`/usb/start?cam_id=${{camId}}&device_index=${{deviceIndex}}`, {{method:'POST'}});
                    const data = await res.json();
                    st.textContent = `✅ ${{camId}} started from /dev/video${{deviceIndex}}`;
                    st.style.color = '#4ade80';
                }} catch(e) {{
                    st.textContent = '❌ Error: ' + e.message;
                    st.style.color = '#ef4444';
                }}
            }}

            async function stopUsbCam(camId) {{
                const st = document.getElementById('usbStatus');
                try {{
                    await fetch(`/usb/stop?cam_id=${{camId}}`, {{method:'POST'}});
                    st.textContent = `⏹️ ${{camId}} stopped`;
                    st.style.color = '#94a3b8';
                }} catch(e) {{}}
            }}

            // ── Status Poller ──
            function sourceIcon(src) {{
                if (!src || src === 'mobile') return '📱';
                if (src.startsWith('usb')) return '🔌';
                return '📷';
            }}

            async function pollStatus() {{
                try {{
                    const res = await fetch('/stream/status');
                    const data = await res.json();
                    const cameras = data.cameras || {{}};
                    const total = data.total_count || 0;
                    const dual = data.dual_fusion || {{}};

                    const cam1 = cameras['cam1'] || {{}};
                    const cam2 = cameras['cam2'] || {{}};

                    // Camera 1 Update
                    const dot1 = document.getElementById('dot1');
                    const badge1 = document.getElementById('badge1');
                    const overlay1 = document.getElementById('overlay1');
                    const count1 = document.getElementById('count1');
                    if (cam1.active) {{
                        dot1.classList.add('live');
                        badge1.textContent = sourceIcon(cam1.source) + ' LIVE (' + cam1.frames + ' frames)';
                        badge1.style.color = '#4ade80';
                        overlay1.classList.add('hidden');
                        count1.textContent = cam1.count;
                    }} else {{
                        dot1.classList.remove('live');
                        badge1.textContent = 'Disconnected';
                        badge1.style.color = '#64748b';
                        count1.textContent = '—';
                    }}

                    // Camera 2 Update
                    const dot2 = document.getElementById('dot2');
                    const badge2 = document.getElementById('badge2');
                    const overlay2 = document.getElementById('overlay2');
                    const count2 = document.getElementById('count2');
                    if (cam2.active) {{
                        dot2.classList.add('live');
                        badge2.textContent = sourceIcon(cam2.source) + ' LIVE (' + cam2.frames + ' frames)';
                        badge2.style.color = '#4ade80';
                        overlay2.classList.add('hidden');
                        count2.textContent = cam2.count;
                    }} else {{
                        dot2.classList.remove('live');
                        badge2.textContent = 'Disconnected';
                        badge2.style.color = '#64748b';
                        count2.textContent = '—';
                    }}

                    // Global Total Banner
                    const globalDot = document.getElementById('globalDot');
                    const globalText = document.getElementById('globalStatusText');
                    const totalNum = document.getElementById('totalCountNum');
                    const totalSub = document.getElementById('totalSubtitle');

                    if (cam1.active || cam2.active) {{
                        globalDot.classList.add('live');
                        if (cam1.active && cam2.active) {{
                            globalText.textContent = '2 Cameras Streaming Live (Layer Fusion Active)';
                            totalSub.textContent = 'Calculated 3D Pallet Total (Front × Side Layer Multiplication)';
                        }} else {{
                            globalText.textContent = '1 Camera Streaming Live';
                            totalSub.textContent = 'Visible carton count on active single camera';
                        }}
                        totalNum.textContent = total;
                    }} else {{
                        globalDot.classList.remove('live');
                        globalText.textContent = 'Waiting for cameras...';
                        totalSub.textContent = 'Combined count across active camera feeds';
                        totalNum.textContent = '—';
                    }}

                    // Layer Breakdown Details Table
                    const layerCard = document.getElementById('layerDetailsCard');
                    const layerTbody = document.getElementById('layerTableBody');
                    const layerBadge = document.getElementById('layerSummaryBadge');

                    if (dual.is_dual_active && dual.layers && dual.layers.length > 0) {{
                        layerCard.style.display = 'block';
                        layerBadge.textContent = `${{dual.layers.length}} Layers Detected`;
                        layerTbody.innerHTML = dual.layers.map(l => `
                            <tr>
                                <td style="font-weight:bold; color:#60a5fa;">Layer ${{l.layer_index}}</td>
                                <td>${{l.front_count}} cartons</td>
                                <td>${{l.side_count}} cartons</td>
                                <td style="color:#94a3b8;">${{l.front_count}} × ${{l.side_count}}</td>
                                <td style="font-weight:900; color:#4ade80; text-align:right;">${{l.layer_total}}</td>
                            </tr>
                        `).join('');
                    }} else {{
                        layerCard.style.display = 'none';
                    }}

                }} catch(e) {{
                    console.error('Status poll error', e);
                }}
            }}

            setInterval(pollStatus, 1500);
            pollStatus();
        </script>
    </body>
    </html>
    """



if __name__ == "__main__":
    import uvicorn

    http_port = int(os.getenv("PORT", "8001"))
    print(f"Starting Carton Counter HTTP server on http://0.0.0.0:{http_port}")
    uvicorn.run(app, host="0.0.0.0", port=http_port, log_level="info")

