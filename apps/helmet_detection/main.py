"""FastAPI app: Helmet Detection (App 2).

Live video from a mobile camera is consumed via frame push:
POST /ingest/frame with JPEG frames, then GET /stream/detect for
annotated live output. API-only — no UI.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from detector import HelmetDetector, PersonStatus
from streamer import FrameBuffer, mjpeg_from_buffer

app = FastAPI(
    title="Helmet Detection",
    description="Detect persons with/without helmets from live camera frames",
    version="1.0.0",
)

_detector: Optional[HelmetDetector] = None
_ingest_buffer = FrameBuffer(max_frames=10)
_violations_log: deque = deque(maxlen=200)
_last_seen_violation: float = 0.0


def _get_detector() -> HelmetDetector:
    global _detector
    if _detector is None:
        _detector = HelmetDetector()
    return _detector


def _read_image(file_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="Invalid image data")
    return img


def _draw(frame: np.ndarray, result) -> np.ndarray:
    vis = frame.copy()
    colors = {"helmet": (0, 200, 0), "no_helmet": (0, 0, 255), "unknown": (0, 165, 255)}
    for p in result.persons:
        x1, y1, x2, y2 = [int(c) for c in p.bbox]
        color = colors.get(p.status, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = p.status.replace("_", " ").upper()
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return vis


@app.on_event("startup")
async def startup():
    try:
        _get_detector()
    except Exception as e:
        print(f"Warning: detector init failed: {e}")


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
async def detect(file: UploadFile = File(...), confidence: float = Query(default=0.5)):
    image = _read_image(await file.read())
    result = _get_detector().detect(image, confidence=confidence)
    return result.to_dict()


@app.post("/ingest/frame")
async def ingest_frame(file: UploadFile = File(...)):
    """Push one JPEG frame from the mobile camera; repeat continuously."""
    image = _read_image(await file.read())
    _ingest_buffer.update(image)
    return {
        "status": "accepted",
        "ingest_frames": _ingest_buffer.frame_count,
        "size": [image.shape[1], image.shape[0]],
    }


@app.post("/ingest/frame/check")
async def ingest_frame_check(confidence: float = Query(default=0.5)):
    """Run detection on the most recent pushed frame and log violations."""
    latest = _ingest_buffer.get_latest()
    if latest is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=409, detail="No frames ingested yet — POST /ingest/frame first")
    _, frame = latest
    result = _get_detector().detect(frame, confidence=confidence)
    ts = time.time()
    for v in result.violations:
        _violations_log.append({"timestamp": ts, **v.to_dict()})
    return result.to_dict()


@app.get("/violations")
async def violations(limit: int = Query(default=50, ge=1, le=200)):
    """Recent no-helmet events seen via /ingest/frame/check."""
    items = list(_violations_log)[-limit:]
    return {"count": len(items), "violations": items}


@app.get("/stream")
async def stream():
    gen = mjpeg_from_buffer(_ingest_buffer)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stream/detect")
async def stream_detect(confidence: float = Query(default=0.5)):
    """Live annotated MJPEG: green=helmet, red=no_helmet, orange=unknown."""

    def annotate(frame: np.ndarray):
        try:
            result = _get_detector().detect(frame, confidence=confidence)
            for v in result.violations:
                _violations_log.append({"timestamp": time.time(), **v.to_dict()})
            return _draw(frame, result)
        except Exception:
            return frame

    gen = mjpeg_from_buffer(_ingest_buffer, quality=75, transform=annotate)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/detect/visualize")
async def detect_visualize(file: UploadFile = File(...), confidence: float = Query(default=0.5)):
    image = _read_image(await file.read())
    result = _get_detector().detect(image, confidence=confidence)
    _, buffer = cv2.imencode(".jpg", _draw(image, result), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Helmet Detection</title></head>
    <body>
        <h1>Helmet Detection API</h1>
        <ul><li><a href="/docs">API Documentation</a></li></ul>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8002")))
