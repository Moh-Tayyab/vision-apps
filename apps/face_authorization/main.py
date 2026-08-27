"""FastAPI app: Authorized/Unauthorized Person Detection (App 3).

Enroll authorized persons' face embeddings (deepface), then compare faces
from live video frames pushed from a mobile camera. API-only — no UI.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import deque
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from face_engine import COSINE_THRESHOLD, DETECTOR_BACKEND, FaceEngine

app = FastAPI(
    title="Face Authorization",
    description="Authorized vs unauthorized person detection using deepface embeddings",
    version="1.0.0",
)

DATA_DIR = os.getenv("DATA_DIR", "data")
_engine: Optional[FaceEngine] = None
_ingest_buffer = None  # FrameBuffer, imported lazily to keep startup light
_events_log: deque = deque(maxlen=300)


def _get_engine() -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine(os.path.join(DATA_DIR, "embeddings.json"))
    return _engine


def _get_buffer():
    global _ingest_buffer
    if _ingest_buffer is None:
        from streamer import FrameBuffer

        _ingest_buffer = FrameBuffer(max_frames=10)
    return _ingest_buffer


def _read_image(file_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    return img


@app.on_event("startup")
async def startup():
    try:
        _get_engine()
    except Exception as e:
        print(f"Warning: engine init failed: {e}")


@app.get("/health")
async def health():
    engine = _get_engine()
    buffer = _get_buffer()
    return {
        "status": "healthy",
        "model_loaded": engine.model_loaded,
        "enrolled_persons": len(engine.list_persons()),
        "ingest_active": buffer.is_active,
        "ingest_frames": buffer.frame_count,
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


# ---------------- Verification ----------------


def _verify_frame(image: np.ndarray) -> dict:
    """Detect faces in the frame and match each against stored embeddings."""
    try:
        from deepface import DeepFace
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"deepface/tensorflow not available in this environment: {e}",
        )

    try:
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            color_face="bgr",
        )
    except Exception:
        faces = []

    results = []
    for i, face in enumerate(faces):
        facial_area = face.get("facial_area", {})
        x, y, w, h = (
            int(facial_area.get("x", 0)),
            int(facial_area.get("y", 0)),
            int(facial_area.get("w", 0)),
            int(facial_area.get("h", 0)),
        )
        confidence = float(face.get("confidence", 0.0))
        # Skip tiny / degenerate detections
        if w < 40 or h < 40 or confidence < 0.5:
            continue
        match = _get_engine().identify_face(face["face"])
        entry = {
            "bbox": [x, y, x + w, y + h],
            "confidence": round(confidence, 3),
        }
        if match is None:
            entry.update(status="unknown", reason="no enrolled persons")
        else:
            entry["matched_name"] = match["name"]
            entry["distance"] = match["distance"]
            entry["status"] = "authorized" if match["authorized"] else "unauthorized"
        results.append(entry)

        if entry["status"] in ("authorized", "unauthorized"):
            _events_log.append({"timestamp": time.time(), **entry})

    return {
        "num_faces": len(results),
        "faces": results,
        "any_unauthorized": any(f["status"] == "unauthorized" for f in results),
    }


@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    """Verify one image against enrolled embeddings."""
    image = _read_image(await file.read())
    return _verify_frame(image)


# ---------------- Live camera (frame push) ----------------


@app.post("/ingest/frame")
async def ingest_frame(file: UploadFile = File(...)):
    """Push one JPEG frame from the mobile camera; repeat continuously."""
    image = _read_image(await file.read())
    buffer = _get_buffer()
    buffer.update(image)
    return {"status": "accepted", "ingest_frames": buffer.frame_count}


@app.post("/ingest/frame/check")
async def ingest_frame_check():
    """Run authorization on the most recent pushed frame; log events."""
    latest = _get_buffer().get_latest()
    if latest is None:
        raise HTTPException(status_code=409, detail="No frames ingested yet — POST /ingest/frame first")
    _, frame = latest
    return _verify_frame(frame)


@app.get("/events")
async def events(limit: int = Query(default=50, ge=1, le=300)):
    items = list(_events_log)[-limit:]
    unauthorized = sum(1 for e in items if e["status"] == "unauthorized")
    return {"count": len(items), "unauthorized_count": unauthorized, "events": items}


@app.get("/stream")
async def stream():
    from streamer import mjpeg_from_buffer

    gen = mjpeg_from_buffer(_get_buffer())
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stream/detect")
async def stream_detect():
    """Live annotated MJPEG: green=AUTHORIZED, red=UNAUTHORIZED."""
    from streamer import mjpeg_from_buffer

    colors = {"authorized": (0, 200, 0), "unauthorized": (0, 0, 255), "unknown": (0, 165, 255)}

    def annotate(frame: np.ndarray):
        try:
            result = _verify_frame(frame)
        except Exception:
            return frame
        vis = frame.copy()
        for f in result["faces"]:
            x1, y1, x2, y2 = f["bbox"]
            color = colors.get(f["status"], (255, 255, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f["matched_name"] if f["status"] == "authorized" else f["status"].upper()
            cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return vis

    gen = mjpeg_from_buffer(_get_buffer(), quality=75, transform=annotate)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Face Authorization</title></head>
    <body>
        <h1>Face Authorization API</h1>
        <ul><li><a href="/docs">API Documentation</a></li></ul>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8003")))
