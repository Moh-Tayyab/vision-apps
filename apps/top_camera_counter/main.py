"""Top Camera Carton Counter - Live Video Tracking.

Real-time carton counting from overhead camera with pick detection.
Tracks cartons with unique IDs and detects when workers remove them.
"""

from __future__ import annotations

import base64
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from detector import CartonDetector, DetectorError
from live_counter import LiveCartonCounter, PalletROI, LiveCountResult

app = FastAPI(
    title="Top Camera Carton Counter",
    description="Live video tracking with pick detection from overhead camera",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector: Optional[CartonDetector] = None
_live_counter: Optional[LiveCartonCounter] = None


def get_detector() -> CartonDetector:
    global _detector
    if _detector is None:
        _detector = CartonDetector()
    return _detector


def get_live_counter() -> LiveCartonCounter:
    global _live_counter
    if _live_counter is None:
        _live_counter = LiveCartonCounter(get_detector())
    return _live_counter


async def _read_image(upload: UploadFile, label: str) -> np.ndarray:
    raw = await upload.read()
    if not raw:
        raise HTTPException(400, f"{label} image is empty")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, f"{label} image could not be decoded")
    return image


@app.get("/health")
async def health():
    ready = bool(os.getenv("ROBOFLOW_API_KEY"))
    return {
        "status": "healthy" if ready else "degraded",
        "detector_configured": ready,
        "mode": "live_tracking",
    }


@app.get("/model/info")
async def model_info():
    try:
        return get_detector().info()
    except DetectorError as exc:
        raise HTTPException(503, str(exc))


@app.get("/live/info")
async def live_info():
    """Get live counter status."""
    counter = get_live_counter()
    return {
        "mode": "live_tracking",
        "initial_cartons": counter.initial_cartons,
        "tracked_cartons": len(counter.state_machine.get_active_cartons()),
        "picked_cartons": counter.state_machine.picked_count,
    }


@app.post("/live/init")
async def init_live_counter(
    initial_count: int = Query(..., ge=1, description="Initial carton count on pallet"),
    roi_x1: int = Query(default=0, description="ROI left x"),
    roi_y1: int = Query(default=0, description="ROI top y"),
    roi_x2: int = Query(default=1920, description="ROI right x"),
    roi_y2: int = Query(default=1080, description="ROI bottom y"),
):
    """Initialize live counter with pallet ROI and initial count."""
    counter = get_live_counter()
    counter.set_initial_count(initial_count)
    counter.pallet_roi = PalletROI(roi_x1, roi_y1, roi_x2, roi_y2)
    counter.reset()
    return {
        "status": "initialized",
        "initial_count": initial_count,
        "roi": {"x1": roi_x1, "y1": roi_y1, "x2": roi_x2, "y2": roi_y2},
    }


@app.post("/live/frame")
async def process_live_frame(
    image: UploadFile = File(..., description="Video frame to process"),
    confidence: float = Query(default=0.36, ge=0.05, le=0.95),
    annotate: bool = Query(default=True, description="Include annotated frame"),
):
    """Process a single video frame for live counting."""
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "Image is empty")
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Image could not be decoded")

    counter = get_live_counter()
    counter.detector.confidence = confidence
    result = counter.process_frame(frame)

    payload = {
        "total_active": result.total_active,
        "total_picked": result.total_picked,
        "layer_counts": result.layer_counts,
        "cartons_by_state": result.cartons_by_state,
        "hand_detected": result.hand_detected,
        "picking_in_progress": result.picking_in_progress,
        "frame_time_ms": result.frame_time_ms,
        "tracks": result.tracks,
        "events": result.events,
    }

    if annotate:
        vis = counter.annotate_frame(frame, result)
        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        payload["annotated_frame"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    return JSONResponse(payload)


@app.websocket("/ws/live")
async def websocket_live_counter(websocket: WebSocket):
    """WebSocket endpoint for live video streaming."""
    await websocket.accept()
    counter = get_live_counter()

    try:
        while True:
            data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            result = counter.process_frame(frame)
            vis = counter.annotate_frame(frame, result)
            _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])

            await websocket.send_bytes(buf.tobytes())

            await websocket.send_json({
                "total_active": result.total_active,
                "total_picked": result.total_picked,
                "hand_detected": result.hand_detected,
                "picking_in_progress": result.picking_in_progress,
                "events": result.events,
            })
    except WebSocketDisconnect:
        pass


@app.post("/live/reset")
async def reset_live_counter():
    """Reset live counter state."""
    counter = get_live_counter()
    counter.reset()
    return {"status": "reset"}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Top Camera Carton Counter</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 24px; line-height: 1.5; }
  .wrap { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 1.6rem; color: #38bdf8; margin-bottom: 4px; }
  .sub { color: #94a3b8; font-size: .9rem; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 18px; }
  @media (max-width: 780px) { .grid { grid-template-columns: 1fr; } }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; }
  .card h3 { font-size: .95rem; color: #7dd3fc; margin-bottom: 10px; }
  .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 18px; }
  button { background: #0284c7; color: #fff; border: 0; border-radius: 8px;
           padding: 12px 26px; font-size: .95rem; font-weight: 600; cursor: pointer; }
  button:disabled { background: #475569; cursor: not-allowed; }
  button.danger { background: #dc2626; }
  label.conf { font-size: .85rem; color: #94a3b8; }
  input[type=range] { vertical-align: middle; }
  #status { margin: 14px 0; font-size: .9rem; color: #94a3b8; min-height: 1.2em; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }
  .stat { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; text-align: center; }
  .stat .n { font-size: 2rem; font-weight: 700; line-height: 1.1; }
  .stat .l { font-size: .75rem; color: #94a3b8; text-transform: uppercase; }
  .stat.active .n { color: #6ee7b7; }
  .stat.picked .n { color: #fbbf24; }
  .stat.hand .n { color: #c084fc; }
  .stat.time .n { color: #38bdf8; }
  .video { width: 100%; border-radius: 10px; border: 1px solid #334155; background: #000; }
  .events { max-height: 200px; overflow-y: auto; font-size: .85rem; }
  .event { padding: 6px 0; border-bottom: 1px solid #334155; }
  .warn { background: #422006; border: 1px solid #a16207; color: #fde68a;
          border-radius: 8px; padding: 10px 14px; font-size: .85rem; margin-bottom: 10px; }
  .err { background: #450a0a; border: 1px solid #dc2626; color: #fecaca;
         border-radius: 8px; padding: 12px 14px; font-size: .9rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Top Camera Carton Counter</h1>
  <p class="sub">Live video tracking with pick detection from overhead camera</p>

  <div class="controls">
    <button id="start">Start Live</button>
    <button id="stop" disabled class="danger">Stop</button>
    <button id="reset">Reset</button>
    <label class="conf">confidence
      <input type="range" id="conf" min="0.05" max="0.95" step="0.01" value="0.36">
      <span id="confVal">0.36</span>
    </label>
  </div>

  <div class="stats">
    <div class="stat active"><div class="n" id="activeN">0</div><div class="l">Active Cartons</div></div>
    <div class="stat picked"><div class="n" id="pickedN">0</div><div class="l">Picked</div></div>
    <div class="stat hand"><div class="n" id="handN">NO</div><div class="l">Hand Detected</div></div>
    <div class="stat time"><div class="n" id="timeN">0ms</div><div class="l">Frame Time</div></div>
  </div>

  <div id="status"></div>

  <div class="card" style="margin-bottom:18px">
    <h3>Live Feed</h3>
    <img class="video" id="video" alt="Live feed will appear here">
  </div>

  <div class="grid">
    <div class="card">
      <h3>Pick Events</h3>
      <div class="events" id="events"></div>
    </div>
    <div class="card">
      <h3>Active Tracks</h3>
      <div class="events" id="tracks"></div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let ws = null;

$('conf').addEventListener('input', e => $('confVal').textContent = e.target.value);

$('start').addEventListener('click', async () => {
  const initialCount = prompt('Initial carton count on pallet:', '24');
  if (!initialCount) return;

  try {
    const res = await fetch('/live/init?initial_count=' + initialCount, { method: 'POST' });
    const data = await res.json();
    $('status').textContent = 'Initialized: ' + data.initial_count + ' cartons';
  } catch (err) {
    $('status').innerHTML = '<div class="err">' + err.message + '</div>';
    return;
  }

  ws = new WebSocket('ws://' + location.host + '/ws/live');
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    $('start').disabled = true;
    $('stop').disabled = false;
    $('status').textContent = 'Connected - streaming live feed';
  };

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      const blob = new Blob([event.data], { type: 'image/jpeg' });
      $('video').src = URL.createObjectURL(blob);
    } else {
      const data = JSON.parse(event.data);
      $('activeN').textContent = data.total_active;
      $('pickedN').textContent = data.total_picked;
      $('handN').textContent = data.hand_detected ? 'YES' : 'NO';
      $('timeN').textContent = Math.round(data.frame_time_ms) + 'ms';

      if (data.events && data.events.length > 0) {
        data.events.forEach(e => {
          const div = document.createElement('div');
          div.className = 'event';
          div.textContent = new Date(e.time * 1000).toLocaleTimeString() + ': ' + e.event + ' (track ' + e.track_id + ')';
          $('events').prepend(div);
        });
      }
    }
  };

  ws.onclose = () => {
    $('start').disabled = false;
    $('stop').disabled = true;
    $('status').textContent = 'Disconnected';
  };
});

$('stop').addEventListener('click', () => {
  if (ws) ws.close();
});

$('reset').addEventListener('click', async () => {
  if (ws) ws.close();
  await fetch('/live/reset', { method: 'POST' });
  $('activeN').textContent = '0';
  $('pickedN').textContent = '0';
  $('handN').textContent = 'NO';
  $('events').innerHTML = '';
  $('status').textContent = 'Reset complete';
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8005"))
    print(f"Starting Top Camera Counter on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
