"""FastAPI app for Carton Counter - pallet counting with YOLO detection."""

from __future__ import annotations

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

app = FastAPI(
    title="Carton Counter",
    description="Pallet carton counting system with multi-angle fusion",
    version="1.1.0",
)

_detector: Optional[CartonDetector] = None
_counter: Optional[CartonCounter] = None
_stream: Optional[MobileCameraStream] = None
_ingest_buffer = FrameBuffer(max_frames=10)


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
    except Exception as e:
        print(f"Warning: detector init failed: {e}")


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
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return vis


@app.post("/detect/visualize")
async def detect_visualize(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.5),
):
    contents = await file.read()
    image = _read_image(contents)

    detector = _get_detector()
    result = detector.detect(image, confidence=confidence)

    vis = _draw_detections(image, result.detections)
    _, buffer = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/count")
async def count(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    top: UploadFile = File(...),
    confidence: float = Query(default=0.5),
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
    """Accept a frame pushed from any camera client (e.g. a phone).

    Push JPEG frames repeatedly from the mobile side (curl / IP webcam app /
    custom client); /stream and /stream/detect then serve this feed.
    """
    image = _read_image(await file.read())
    _ingest_buffer.update(image)
    return {
        "status": "accepted",
        "ingest_frames": _ingest_buffer.frame_count,
        "size": [image.shape[1], image.shape[0]],
    }


def _live_source() -> str:
    return "ingest" if _ingest_buffer.is_active else "local_camera"


@app.get("/stream")
async def stream():
    """MJPEG live view. Serves pushed frames when available, else local camera."""
    if _ingest_buffer.is_active:
        gen = mjpeg_from_buffer(_ingest_buffer)
    else:
        gen = _ensure_started(_get_stream()).mjpeg_generator()
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/detect")
async def stream_detect(confidence: float = Query(default=0.5)):
    """MJPEG stream with live YOLO boxes drawn on every frame."""

    def annotate(frame: np.ndarray) -> np.ndarray:
        try:
            result = _get_detector().detect(frame, confidence=confidence)
            return _draw_detections(frame, result.detections)
        except Exception:
            return frame

    if _ingest_buffer.is_active:
        gen = mjpeg_from_buffer(_ingest_buffer, quality=75, transform=annotate)
    else:
        stream_obj = _ensure_started(_get_stream())
        buffer = stream_obj._buffer
        gen = mjpeg_from_buffer(buffer, quality=75, transform=annotate)
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


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


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Carton Counter</title></head>
    <body>
        <h1>Carton Counter API</h1>
        <ul>
            <li><a href="/docs">API Documentation</a></li>
            <li><a href="/health">Health Check</a></li>
        </ul>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
