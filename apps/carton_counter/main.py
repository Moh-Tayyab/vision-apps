"""FastAPI app for Carton Counter - pallet counting with YOLO detection."""

from __future__ import annotations

import io
import os
import time
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response

from detector import CartonDetector
from counter import CartonCounter, CountResult
from streamer import MobileCameraStream, websocket_stream

app = FastAPI(
    title="Carton Counter",
    description="Pallet carton counting system with multi-angle fusion",
    version="1.0.0",
)

_detector: Optional[CartonDetector] = None
_counter: Optional[CartonCounter] = None
_stream: Optional[MobileCameraStream] = None


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
    result = detector.detect(image)
    return result.to_dict()


@app.post("/detect/visualize")
async def detect_visualize(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.5),
):
    contents = await file.read()
    image = _read_image(contents)

    detector = _get_detector()
    result = detector.detect(image)

    vis = image.copy()
    for det in result.detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    _, buffer = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/count")
async def count(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    top: UploadFile = File(...),
):
    counter = _get_counter()
    images = []
    for f in [front, side, top]:
        contents = await f.read()
        images.append(_read_image(contents))

    result = counter.count_multi_angle(images)
    return result.to_dict()


@app.post("/count/video")
async def count_video(
    file: UploadFile = File(...),
    sample_frames: int = Query(default=5, ge=1, le=30),
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(nparr, cv2.IMREAD_COLOR))
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

    counter = _get_counter()
    result = counter.count_multi_frame(frames)
    return result.to_dict()


@app.get("/stream")
async def stream():
    stream_obj = _get_stream()
    if not stream_obj.is_active:
        stream_obj.start()
    return StreamingResponse(
        stream_obj.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


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
            <li><a href="/stream">Live Camera Stream</a></li>
            <li><a href="/health">Health Check</a></li>
        </ul>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
