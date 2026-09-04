"""Row-wise pallet carton counter.

Takes two still photos of a pallet from adjacent faces 90 degrees apart and
reports the carton count row by row. Opposite faces show the same dimension,
so adjacent views are what give both grid dimensions:

    row total    = faces in front row x faces in side row
    pallet total = sum of row totals

Images only -- there is no live video path in this app by design.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from detector import CartonFaceDetector, DetectorError
from row_engine import count_pallet
from visualizer import annotate_pair, build_summary_panel, to_data_uri

app = FastAPI(
    title="Pallet Counter (Row-Wise)",
    description="Two-image row-by-row pallet carton counting",
    version="1.0.0",
)

_detector: Optional[CartonFaceDetector] = None


def get_detector() -> CartonFaceDetector:
    global _detector
    if _detector is None:
        _detector = CartonFaceDetector()
    return _detector


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
        "mode": "two_image_row_wise",
    }


@app.get("/model/info")
async def model_info():
    try:
        return get_detector().info()
    except DetectorError as exc:
        raise HTTPException(503, str(exc))


@app.post("/count/pallet")
async def count(
    front: UploadFile = File(..., description="View of one face"),
    side: UploadFile = File(..., description="View of the adjacent face, 90 degrees around"),
    confidence: float = Query(default=0.36, ge=0.05, le=0.95),
    annotate: bool = Query(default=True, description="Include annotated images"),
):
    """Count a pallet from two adjacent-face photos."""
    front_img = await _read_image(front, "front")
    side_img = await _read_image(side, "side")

    started = time.perf_counter()
    try:
        detector = get_detector()
        front_boxes, _ = detector.detect(front_img, confidence)
        side_boxes, _ = detector.detect(side_img, confidence)
    except DetectorError as exc:
        raise HTTPException(503, str(exc))

    fh, fw = front_img.shape[:2]
    sh, sw = side_img.shape[:2]
    result = count_pallet(
        front_boxes, (fw, fh),
        side_boxes, (sw, sh),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )

    payload = result.to_dict()
    if annotate:
        front_vis, side_vis = annotate_pair(front_img, side_img, result)
        payload["front_annotated"] = to_data_uri(front_vis)
        payload["side_annotated"] = to_data_uri(side_vis)
        payload["summary_panel"] = to_data_uri(build_summary_panel(result))
    return JSONResponse(payload)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pallet Counter - Row Wise</title>
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
  input[type=file] { width: 100%; color: #cbd5e1; font-size: .85rem; }
  .preview { width: 100%; margin-top: 10px; border-radius: 8px; display: none; }
  .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 18px; }
  button { background: #0284c7; color: #fff; border: 0; border-radius: 8px;
           padding: 12px 26px; font-size: .95rem; font-weight: 600; cursor: pointer; }
  button:disabled { background: #475569; cursor: not-allowed; }
  label.conf { font-size: .85rem; color: #94a3b8; }
  input[type=range] { vertical-align: middle; }
  #status { margin: 14px 0; font-size: .9rem; color: #94a3b8; min-height: 1.2em; }
  .total { background: #064e3b; border: 1px solid #059669; border-radius: 12px;
           padding: 18px; text-align: center; margin-bottom: 18px; display: none; }
  .total .n { font-size: 3rem; font-weight: 700; color: #6ee7b7; line-height: 1.1; }
  .total .l { font-size: .85rem; color: #a7f3d0; text-transform: uppercase; letter-spacing: .08em; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: .9rem; }
  th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid #334155; }
  th { color: #94a3b8; font-weight: 600; font-size: .78rem; text-transform: uppercase; }
  td.n { font-variant-numeric: tabular-nums; }
  .warn { background: #422006; border: 1px solid #a16207; color: #fde68a;
          border-radius: 8px; padding: 10px 14px; font-size: .85rem; margin-bottom: 10px; }
  .err { background: #450a0a; border: 1px solid #dc2626; color: #fecaca;
         border-radius: 8px; padding: 12px 14px; font-size: .9rem; }
  .out { display: none; }
  .out img { width: 100%; border-radius: 10px; border: 1px solid #334155; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Pallet Counter &mdash; Row Wise</h1>
  <p class="sub">Two photos of adjacent faces, 90&deg; apart. Each row is counted as
     front &times; side, then summed.</p>

  <div class="grid">
    <div class="card">
      <h3>View A &mdash; one face</h3>
      <input type="file" id="frontFile" accept="image/*">
      <img class="preview" id="frontPrev">
    </div>
    <div class="card">
      <h3>View B &mdash; adjacent face (90&deg;)</h3>
      <input type="file" id="sideFile" accept="image/*">
      <img class="preview" id="sidePrev">
    </div>
  </div>

  <div class="controls">
    <button id="run" disabled>Count Pallet</button>
    <label class="conf">confidence
      <input type="range" id="conf" min="0.05" max="0.95" step="0.01" value="0.36">
      <span id="confVal">0.36</span>
    </label>
  </div>

  <div id="status"></div>

  <div class="total" id="total">
    <div class="n" id="totalN">0</div>
    <div class="l">cartons on pallet</div>
  </div>

  <div id="warnings"></div>

  <div class="out" id="out">
    <div class="card" style="margin-bottom:16px">
      <h3>Row breakdown</h3>
      <table>
        <thead><tr><th>Row</th><th>View A</th><th>View B</th><th>Cartons</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <div class="grid">
      <div class="card"><h3>View A &mdash; rows detected</h3><img id="frontOut"></div>
      <div class="card"><h3>View B &mdash; rows detected</h3><img id="sideOut"></div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const files = { front: null, side: null };

function hook(inputId, prevId, key) {
  $(inputId).addEventListener('change', e => {
    const f = e.target.files[0];
    files[key] = f || null;
    const img = $(prevId);
    if (f) { img.src = URL.createObjectURL(f); img.style.display = 'block'; }
    else { img.style.display = 'none'; }
    $('run').disabled = !(files.front && files.side);
  });
}
hook('frontFile', 'frontPrev', 'front');
hook('sideFile', 'sidePrev', 'side');

$('conf').addEventListener('input', e => $('confVal').textContent = e.target.value);

$('run').addEventListener('click', async () => {
  $('run').disabled = true;
  $('status').textContent = 'Detecting cartons in both views...';
  $('out').style.display = 'none';
  $('total').style.display = 'none';
  $('warnings').innerHTML = '';

  const fd = new FormData();
  fd.append('front', files.front);
  fd.append('side', files.side);

  try {
    const res = await fetch('/count/pallet?confidence=' + $('conf').value, {
      method: 'POST', body: fd
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'request failed');

    $('totalN').textContent = data.total_count;
    $('total').style.display = 'block';

    $('rows').innerHTML = data.rows.map(r =>
      `<tr><td class="n">${r.row}</td><td class="n">${r.front_count}</td>` +
      `<td class="n">${r.side_count}</td><td class="n"><b>${r.row_total}</b></td></tr>`
    ).join('');

    $('warnings').innerHTML = (data.warnings || [])
      .map(w => `<div class="warn">${w}</div>`).join('');

    $('frontOut').src = data.front_annotated;
    $('sideOut').src = data.side_annotated;
    $('out').style.display = 'block';
    $('status').textContent =
      `${data.rows_counted} rows - ${data.front_view.total_faces} + ` +
      `${data.side_view.total_faces} faces detected - ` +
      `${Math.round(data.processing_time_ms)} ms`;
  } catch (err) {
    $('status').textContent = '';
    $('warnings').innerHTML = `<div class="err">${err.message}</div>`;
  } finally {
    $('run').disabled = false;
  }
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8004"))
    print(f"Starting Pallet Counter on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
