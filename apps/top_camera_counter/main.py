"""Top Camera & Angled View Carton Counter - Web Application & API.

Processes single angled pallet images (capturing side view rows + top view cartons)
to calculate total rows, top row cartons, and projected total pallet cartons.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from angle_view_counter import (
    analyze_angled_view,
    annotate_angled_view,
    AngledViewResult,
)

app = FastAPI(
    title="PalletVision Angled Carton Counter",
    description="Single-image pallet carton counting from angled/dual-perspective view",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Optional[YOLO] = None
DEFAULT_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "best.pt")
DATASET_IMAGES_DIR = Path(__file__).parent / "dataset" / "images"


def get_yolo_model() -> YOLO:
    """Lazy load YOLO model singleton."""
    global _model
    if _model is None:
        if not Path(DEFAULT_MODEL_PATH).exists():
            raise HTTPException(503, f"YOLO model file not found: {DEFAULT_MODEL_PATH}")
        _model = YOLO(DEFAULT_MODEL_PATH)
    return _model


def decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to OpenCV BGR image."""
    if not raw_bytes:
        raise HTTPException(400, "Uploaded image data is empty")
    image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode uploaded image format")
    return image


@app.get("/health")
async def health():
    ready = Path(DEFAULT_MODEL_PATH).exists()
    return {
        "status": "healthy" if ready else "degraded",
        "model_loaded": _model is not None,
        "model_path": DEFAULT_MODEL_PATH,
        "mode": "single_image_angled_counter",
    }


@app.get("/model/info")
async def model_info():
    ready = Path(DEFAULT_MODEL_PATH).exists()
    if not ready:
        raise HTTPException(503, "Model not configured")
    model = get_yolo_model()
    return {
        "backend": "ultralytics_yolo",
        "model_path": DEFAULT_MODEL_PATH,
        "classes": getattr(model, "names", {0: "carton"}),
    }


@app.post("/api/analyze-image")
async def analyze_image_endpoint(
    image: UploadFile = File(..., description="Angled pallet image file (JPG/PNG)"),
    confidence: float = Query(default=0.36, ge=0.05, le=0.95, description="Detection confidence"),
    overlap_threshold: float = Query(default=0.30, ge=0.05, le=0.90, description="Column grouping overlap"),
    nms_iou: float = Query(default=0.35, ge=0.05, le=0.90, description="NMS IoU threshold"),
    show_top_highlight: bool = Query(default=True, description="Highlight top layer cartons"),
    show_column_colors: bool = Query(default=True, description="Color code column stacks"),
    show_formula_banner: bool = Query(default=True, description="Draw calculation banner overlay"),
):
    """Analyze single angled image to count rows and top cartons."""
    raw = await image.read()
    img = decode_image_bytes(raw)
    h, w = img.shape[:2]

    model = get_yolo_model()
    result: AngledViewResult = analyze_angled_view(
        image=img,
        model=model,
        confidence=confidence,
        overlap_threshold=overlap_threshold,
        nms_iou=nms_iou,
    )

    # Annotate image
    vis = annotate_angled_view(
        image=img,
        result=result,
        show_top_highlight=show_top_highlight,
        show_column_colors=show_column_colors,
        show_formula_banner=show_formula_banner,
    )

    # Encode annotated image to JPEG base64
    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    base64_annotated = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

    # Encode original image thumbnail/base64 for side-by-side comparison
    _, orig_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    base64_original = "data:image/jpeg;base64," + base64.b64encode(orig_buf.tobytes()).decode("utf-8")

    response_data = result.to_dict()
    response_data["image_dimensions"] = {"width": w, "height": h}
    response_data["annotated_image"] = base64_annotated
    response_data["original_image"] = base64_original
    response_data["filename"] = image.filename or "uploaded_image.jpg"

    return JSONResponse(response_data)


@app.get("/api/samples")
async def list_sample_images():
    """List sample images available in the dataset directory."""
    if not DATASET_IMAGES_DIR.exists():
        return {"samples": []}

    samples = []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    for path in sorted(DATASET_IMAGES_DIR.glob("*")):
        if path.suffix.lower() in valid_exts and not path.name.startswith("."):
            samples.append({
                "filename": path.name,
                "size_kb": round(path.stat().st_size / 1024, 1),
                "url": f"/api/sample-image/{path.name}",
            })

    return {"samples": samples}


@app.get("/api/sample-image/{filename}")
async def get_sample_image(filename: str):
    """Serve a sample image from the dataset directory."""
    file_path = DATASET_IMAGES_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "Sample image not found")
    return FileResponse(file_path)


@app.post("/api/analyze-sample")
async def analyze_sample_endpoint(
    filename: str = Query(..., description="Filename of sample image"),
    confidence: float = Query(default=0.36, ge=0.05, le=0.95),
    overlap_threshold: float = Query(default=0.30, ge=0.05, le=0.90),
    nms_iou: float = Query(default=0.35, ge=0.05, le=0.90),
    show_top_highlight: bool = Query(default=True),
    show_column_colors: bool = Query(default=True),
    show_formula_banner: bool = Query(default=True),
):
    """Analyze a pre-loaded sample image from the dataset folder."""
    file_path = DATASET_IMAGES_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, f"Sample image '{filename}' not found")

    img = cv2.imread(str(file_path))
    if img is None:
        raise HTTPException(500, "Could not read sample image")

    h, w = img.shape[:2]
    model = get_yolo_model()
    result = analyze_angled_view(
        image=img,
        model=model,
        confidence=confidence,
        overlap_threshold=overlap_threshold,
        nms_iou=nms_iou,
    )

    vis = annotate_angled_view(
        image=img,
        result=result,
        show_top_highlight=show_top_highlight,
        show_column_colors=show_column_colors,
        show_formula_banner=show_formula_banner,
    )

    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    base64_annotated = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

    _, orig_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    base64_original = "data:image/jpeg;base64," + base64.b64encode(orig_buf.tobytes()).decode("utf-8")

    response_data = result.to_dict()
    response_data["image_dimensions"] = {"width": w, "height": h}
    response_data["annotated_image"] = base64_annotated
    response_data["original_image"] = base64_original
    response_data["filename"] = filename

    return JSONResponse(response_data)


# Backward compatibility for legacy test endpoints
@app.post("/live/frame")
async def legacy_frame_endpoint(
    image: UploadFile = File(...),
    confidence: float = Query(default=0.36),
):
    """Legacy compatibility endpoint."""
    raw = await image.read()
    img = decode_image_bytes(raw)
    model = get_yolo_model()
    result = analyze_angled_view(img, model, confidence=confidence)
    vis = annotate_angled_view(img, result)
    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return JSONResponse({
        "total_active": result.top_row_cartons,
        "top_row_count": result.top_row_cartons,
        "total_rows": result.total_rows,
        "total_picked": 0,
        "annotated_frame": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8"),
    })


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PalletVision — Angled View Carton Counter</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-base: #080d1a;
    --bg-surface: #0f172a;
    --bg-card: #141f36;
    --bg-card-hover: #1a2846;
    --border: #1e2e4f;
    --border-light: #2c426e;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --primary: #38bdf8;
    --primary-glow: rgba(56, 189, 248, 0.25);
    --emerald: #10b981;
    --emerald-glow: rgba(16, 185, 129, 0.25);
    --amber: #f59e0b;
    --amber-glow: rgba(245, 158, 11, 0.25);
    --purple: #a855f7;
    --purple-glow: rgba(168, 85, 247, 0.25);
    --rose: #f43f5e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: radial-gradient(circle at 50% 0%, #15223e 0%, var(--bg-base) 70%);
    color: var(--text-main);
    min-height: 100vh;
    padding: 24px 20px 40px;
    line-height: 1.5;
  }
  .container { max-width: 1440px; margin: 0 auto; }

  /* Header */
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 20px;
    margin-bottom: 24px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
    gap: 16px;
  }
  .brand-group { display: flex; align-items: center; gap: 14px; }
  .logo-icon {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, #0284c7, #38bdf8);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 20px var(--primary-glow);
  }
  .logo-icon svg { width: 26px; height: 26px; stroke: #fff; fill: none; stroke-width: 2; }
  h1 { font-size: 1.55rem; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
  .tagline { color: var(--text-muted); font-size: 0.88rem; margin-top: 2px; }
  .header-badges { display: flex; align-items: center; gap: 10px; }
  .badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 9999px;
    font-size: 0.82rem; font-weight: 600;
  }
  .badge-healthy {
    color: #34d399; background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
  }
  .badge-dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981; }

  /* Layout Grid */
  .main-grid {
    display: grid;
    grid-template-columns: 440px 1fr;
    gap: 24px;
  }
  @media (max-width: 1100px) {
    .main-grid { grid-template-columns: 1fr; }
  }

  /* Cards */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
  }
  .card-title {
    font-size: 0.95rem; font-weight: 700;
    color: var(--primary); margin-bottom: 14px;
    display: flex; align-items: center; justify-content: space-between;
    letter-spacing: 0.3px; text-transform: uppercase;
  }

  /* Upload Box */
  .drop-zone {
    border: 2px dashed var(--border-light);
    border-radius: 14px;
    padding: 28px 20px;
    text-align: center;
    cursor: pointer;
    background: rgba(15, 23, 42, 0.6);
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
  }
  .drop-zone:hover, .drop-zone.dragover {
    border-color: var(--primary);
    background: rgba(56, 189, 248, 0.08);
    box-shadow: 0 0 24px var(--primary-glow);
  }
  .drop-icon {
    width: 48px; height: 48px; margin: 0 auto 12px;
    background: rgba(56, 189, 248, 0.12);
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
  }
  .drop-icon svg { width: 24px; height: 24px; stroke: var(--primary); fill: none; stroke-width: 2; }
  .drop-title { font-weight: 700; font-size: 1rem; color: #fff; margin-bottom: 4px; }
  .drop-sub { font-size: 0.82rem; color: var(--text-muted); }

  /* Sample Gallery */
  .samples-title { font-size: 0.82rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; margin: 16px 0 8px; }
  .samples-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    max-height: 180px;
    overflow-y: auto;
    padding-right: 4px;
  }
  .sample-card {
    background: #0f172a; border: 1px solid var(--border);
    border-radius: 10px; padding: 6px; cursor: pointer;
    transition: all 0.2s ease; text-align: center;
  }
  .sample-card:hover, .sample-card.active {
    border-color: var(--primary);
    background: rgba(56, 189, 248, 0.12);
    transform: translateY(-2px);
  }
  .sample-thumb {
    width: 100%; height: 50px; object-fit: cover;
    border-radius: 6px; margin-bottom: 4px; background: #000;
  }
  .sample-name {
    font-size: 0.72rem; color: var(--text-muted);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }

  /* Sliders & Controls */
  .control-group { margin-top: 14px; }
  .control-label {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.85rem; color: var(--text-muted); font-weight: 600; margin-bottom: 6px;
  }
  .control-val { color: var(--primary); font-family: 'JetBrains Mono', monospace; font-weight: 700; }
  input[type=range] {
    width: 100%; height: 6px; border-radius: 4px;
    background: #1e293b; outline: none; -webkit-appearance: none; cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%;
    background: var(--primary); cursor: pointer; box-shadow: 0 0 10px var(--primary);
  }
  .toggles { display: flex; flex-direction: column; gap: 8px; margin-top: 14px; }
  .toggle-item {
    display: flex; align-items: center; gap: 10px;
    font-size: 0.84rem; color: var(--text-muted); cursor: pointer;
  }
  .toggle-item input[type=checkbox] {
    width: 16px; height: 16px; accent-color: var(--emerald); cursor: pointer;
  }

  /* Buttons */
  .btn-row { display: flex; gap: 10px; margin-top: 18px; }
  button {
    font-family: inherit; font-size: 0.9rem; font-weight: 700;
    padding: 12px 20px; border-radius: 10px; border: none;
    cursor: pointer; transition: all 0.2s ease;
    display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  }
  .btn-primary {
    background: linear-gradient(135deg, #0284c7, #38bdf8);
    color: #041329; flex: 2; box-shadow: 0 4px 18px var(--primary-glow);
  }
  .btn-primary:hover {
    background: linear-gradient(135deg, #0369a1, #0284c7);
    color: #fff; transform: translateY(-1px);
  }
  .btn-secondary {
    background: #1e293b; color: var(--text-main);
    border: 1px solid var(--border); flex: 1;
  }
  .btn-secondary:hover { background: #27354f; }
  .btn-camera {
    background: rgba(16, 185, 129, 0.15); color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3); width: 100%; margin-top: 10px;
  }
  .btn-camera:hover { background: rgba(16, 185, 129, 0.25); color: #fff; }

  /* KPI Cards Grid */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 20px;
  }
  @media (max-width: 800px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
  .kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  }
  .kpi-rows::before { background: var(--emerald); box-shadow: 0 0 10px var(--emerald); }
  .kpi-top::before { background: var(--primary); box-shadow: 0 0 10px var(--primary); }
  .kpi-est::before { background: var(--amber); box-shadow: 0 0 12px var(--amber); }
  .kpi-det::before { background: var(--purple); box-shadow: 0 0 10px var(--purple); }

  .kpi-num {
    font-size: 2.6rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;
    line-height: 1.1; margin-bottom: 4px;
  }
  .kpi-rows .kpi-num { color: #34d399; }
  .kpi-top .kpi-num { color: #38bdf8; }
  .kpi-est .kpi-num { color: #fbbf24; text-shadow: 0 0 20px var(--amber-glow); }
  .kpi-det .kpi-num { color: #c084fc; }

  .kpi-label { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px; }
  .kpi-sub { font-size: 0.74rem; color: var(--text-dim); margin-top: 2px; }

  /* Formula Card */
  .formula-card {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
  }
  .formula-badge {
    background: rgba(245, 158, 11, 0.2); color: #fbbf24;
    padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .formula-eq {
    font-size: 1.15rem; font-weight: 700; color: #fff;
    font-family: 'JetBrains Mono', monospace;
  }
  .formula-highlight { color: #fbbf24; font-weight: 800; }

  /* Visualizer Box */
  .visualizer-card { padding: 18px; }
  .view-controls {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px; flex-wrap: wrap; gap: 8px;
  }
  .view-tabs { display: flex; gap: 6px; background: #0f172a; padding: 4px; border-radius: 8px; border: 1px solid var(--border); }
  .view-tab {
    padding: 6px 14px; border-radius: 6px; font-size: 0.82rem; font-weight: 600;
    color: var(--text-muted); cursor: pointer; border: none; background: transparent;
  }
  .view-tab.active { background: var(--primary); color: #041329; font-weight: 700; }
  .img-viewport {
    background: #000;
    border-radius: 12px;
    border: 1px solid var(--border);
    overflow: hidden;
    position: relative;
    min-height: 420px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .main-image {
    max-width: 100%; max-height: 600px;
    width: auto; height: auto;
    object-fit: contain; display: block; border-radius: 8px;
    transition: opacity 0.2s ease;
  }
  .empty-state {
    text-align: center; color: var(--text-dim); padding: 40px 20px;
  }
  .empty-state svg { width: 56px; height: 56px; stroke: var(--border-light); stroke-width: 1.5; margin-bottom: 12px; }

  /* Details Accordion & Breakdown */
  .breakdown-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 10px;
    margin-top: 12px;
  }
  .col-stat-pill {
    background: #0f172a; border: 1px solid var(--border);
    border-radius: 10px; padding: 10px; text-align: center;
  }
  .col-pill-title { font-size: 0.75rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }
  .col-pill-num { font-size: 1.25rem; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono', monospace; }

  /* Camera Modal */
  .modal-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.85);
    display: none; align-items: center; justify-content: center;
    z-index: 9999; backdrop-filter: blur(8px);
  }
  .modal-overlay.open { display: flex; }
  .modal-content {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 18px; padding: 24px; max-width: 640px; width: 90%;
    box-shadow: 0 20px 50px rgba(0,0,0,0.6);
  }
  .cam-video { width: 100%; border-radius: 12px; background: #000; margin-bottom: 16px; aspect-ratio: 4/3; }

  /* Loading Spinner */
  .spinner {
    width: 36px; height: 36px; border: 3px solid rgba(56, 189, 248, 0.2);
    border-top-color: var(--primary); border-radius: 50%;
    animation: spin 0.8s linear infinite; margin: 0 auto 12px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="brand-group">
      <div class="logo-icon">
        <svg viewBox="0 0 24 24"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
      </div>
      <div>
        <h1>PalletVision AI</h1>
        <p class="tagline">Angled View Pallet Counter — Side Rows & Top Cartons in a Single Image</p>
      </div>
    </div>
    <div class="header-badges">
      <span id="modelBadge" class="badge badge-healthy"><span class="badge-dot"></span> YOLO Model Ready</span>
      <span id="perfBadge" class="badge" style="background: rgba(255,255,255,0.06); color: var(--text-muted);"><span id="perfText">Ready</span></span>
    </div>
  </header>

  <!-- KPI Row -->
  <div class="kpi-grid">
    <div class="kpi-card kpi-rows">
      <div class="kpi-num" id="statRows">0</div>
      <div class="kpi-label">Total Rows / Layers</div>
      <div class="kpi-sub">From Side View Stacks</div>
    </div>
    <div class="kpi-card kpi-top">
      <div class="kpi-num" id="statTop">0</div>
      <div class="kpi-label">Top Row Cartons</div>
      <div class="kpi-sub">Visible on Top Layer</div>
    </div>
    <div class="kpi-card kpi-est">
      <div class="kpi-num" id="statEst">0</div>
      <div class="kpi-label">Estimated Pallet Total</div>
      <div class="kpi-sub">Top Cartons × Total Rows</div>
    </div>
    <div class="kpi-card kpi-det">
      <div class="kpi-num" id="statDet">0</div>
      <div class="kpi-label">Visible Detected</div>
      <div class="kpi-sub">All Detected Faces</div>
    </div>
  </div>

  <!-- Formula Highlight -->
  <div class="formula-card">
    <div style="display: flex; align-items: center; gap: 10px;">
      <span class="formula-badge">Calculation Model</span>
      <span class="formula-eq" id="formulaDisplay">Total Estimated = Top Cartons (0) × Total Rows (0) = 0 Cartons</span>
    </div>
    <div style="font-size: 0.85rem; color: var(--text-muted);" id="columnSummaryText">
      Columns Detected: 0
    </div>
  </div>

  <div class="main-grid">
    <!-- Left Column: Inputs & Controls -->
    <div>
      <div class="card">
        <div class="card-title">
          <span>Image Input</span>
          <span style="font-size: 0.75rem; color: var(--text-muted);">Angled View</span>
        </div>

        <div class="drop-zone" id="dropZone">
          <div class="drop-icon">
            <svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
          </div>
          <div class="drop-title">Drop pallet image here</div>
          <div class="drop-sub">or click to browse from device (JPG, PNG)</div>
          <input type="file" id="fileInput" accept="image/*" style="display: none;">
        </div>

        <button class="btn-camera" id="btnCameraModal">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
          Capture with Camera
        </button>

        <div class="samples-title">Or Try Sample Pallet Images:</div>
        <div class="samples-grid" id="samplesGrid">
          <div style="grid-column: span 3; color: var(--text-dim); font-size: 0.8rem; padding: 8px;">Loading samples...</div>
        </div>
      </div>

      <!-- Detection Parameters -->
      <div class="card">
        <div class="card-title">
          <span>Detection Settings</span>
        </div>

        <div class="control-group">
          <div class="control-label">
            <span>Confidence Threshold</span>
            <span class="control-val" id="confVal">0.36</span>
          </div>
          <input type="range" id="confSlider" min="0.10" max="0.90" step="0.02" value="0.36">
        </div>

        <div class="control-group">
          <div class="control-label">
            <span>Column Overlap Threshold</span>
            <span class="control-val" id="overlapVal">0.30</span>
          </div>
          <input type="range" id="overlapSlider" min="0.10" max="0.80" step="0.02" value="0.30">
        </div>

        <div class="control-group">
          <div class="control-label">
            <span>NMS IoU Deduplication</span>
            <span class="control-val" id="nmsVal">0.35</span>
          </div>
          <input type="range" id="nmsSlider" min="0.10" max="0.80" step="0.05" value="0.35">
        </div>

        <div class="toggles">
          <label class="toggle-item">
            <input type="checkbox" id="chkTopHighlight" checked>
            <span>Highlight Top Layer Cartons (Emerald Box + Badges)</span>
          </label>
          <label class="toggle-item">
            <input type="checkbox" id="chkColColors" checked>
            <span>Color-code Vertical Column Stacks</span>
          </label>
          <label class="toggle-item">
            <input type="checkbox" id="chkBanner" checked>
            <span>Embed HUD Banner & Formula on Image</span>
          </label>
        </div>

        <div class="btn-row">
          <button class="btn-primary" id="btnAnalyze">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
            Re-Analyze
          </button>
          <button class="btn-secondary" id="btnReset">Reset</button>
        </div>
      </div>
    </div>

    <!-- Right Column: Visualizer & Breakdown -->
    <div>
      <div class="card visualizer-card">
        <div class="view-controls">
          <div class="view-tabs">
            <button class="view-tab active" id="tabAnnotated">Annotated View</button>
            <button class="view-tab" id="tabOriginal">Original Image</button>
          </div>
          <div style="display: flex; gap: 8px;">
            <button class="btn-secondary" id="btnDownload" style="padding: 6px 14px; font-size: 0.8rem;" disabled>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
              Download
            </button>
            <button class="btn-secondary" id="btnExportJson" style="padding: 6px 14px; font-size: 0.8rem;" disabled>
              JSON
            </button>
          </div>
        </div>

        <div class="img-viewport" id="viewport">
          <div class="empty-state" id="emptyState">
            <svg viewBox="0 0 24 24" fill="none"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
            <p style="font-weight: 600; color: var(--text-muted); font-size: 1rem;">No pallet image analyzed yet</p>
            <p style="font-size: 0.85rem; margin-top: 4px;">Upload an image on the left or select a sample</p>
          </div>
          <img id="mainImg" class="main-image" style="display: none;" alt="Pallet Visualization">
        </div>
      </div>

      <!-- Column Breakdown Card -->
      <div class="card">
        <div class="card-title">
          <span>Vertical Columns & Stack Breakdown</span>
          <span id="colDetailCount" style="font-size: 0.8rem; color: var(--text-muted);">0 Columns</span>
        </div>
        <div class="breakdown-grid" id="breakdownGrid">
          <div style="color: var(--text-dim); font-size: 0.85rem; padding: 8px;">No breakdown data available yet.</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Camera Snapshot Modal -->
<div class="modal-overlay" id="camModal">
  <div class="modal-content">
    <div class="card-title" style="margin-bottom: 16px;">
      <span>Take Photo with Camera</span>
      <button id="closeCamModal" style="background: transparent; color: var(--text-muted); font-size: 1.2rem; padding: 0;">✕</button>
    </div>
    <video id="camVideo" class="cam-video" autoplay playsinline></video>
    <div style="display: flex; gap: 10px;">
      <button class="btn-primary" id="btnSnapPhoto">📸 Snap & Count</button>
      <button class="btn-secondary" id="btnCancelCam">Cancel</button>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let currentFile = null;
let currentSampleName = null;
let lastAnalysisResult = null;
let currentViewMode = 'annotated';
let streamObj = null;

// Initial check
async function initApp() {
  loadSamples();
  checkHealth();
}

async function checkHealth() {
  try {
    const res = await fetch('/health');
    const d = await res.json();
    if (d.status === 'healthy') {
      $('modelBadge').className = 'badge badge-healthy';
      $('modelBadge').innerHTML = '<span class="badge-dot"></span> YOLO Model Ready';
    } else {
      $('modelBadge').className = 'badge';
      $('modelBadge').style.background = 'rgba(239,68,68,0.15)';
      $('modelBadge').style.color = '#f87171';
      $('modelBadge').innerHTML = 'Model Missing (best.pt)';
    }
  } catch(e) {}
}

async function loadSamples() {
  try {
    const res = await fetch('/api/samples');
    const d = await res.json();
    const grid = $('samplesGrid');
    if (!d.samples || d.samples.length === 0) {
      grid.innerHTML = '<div style="grid-column:span 3; color:var(--text-dim); font-size:0.8rem;">No samples found.</div>';
      return;
    }
    grid.innerHTML = '';
    d.samples.forEach((s, idx) => {
      const card = document.createElement('div');
      card.className = 'sample-card';
      card.innerHTML = `
        <img class="sample-thumb" src="${s.url}" alt="${s.filename}" loading="lazy">
        <div class="sample-name" title="${s.filename}">Sample ${idx + 1}</div>
      `;
      card.onclick = () => selectSample(s.filename, card);
      grid.appendChild(card);
    });
  } catch (err) {
    $('samplesGrid').innerHTML = '<div style="color:var(--text-dim); font-size:0.8rem;">Error loading samples.</div>';
  }
}

function selectSample(filename, cardElem) {
  document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
  if (cardElem) cardElem.classList.add('active');
  currentSampleName = filename;
  currentFile = null;
  runAnalysis();
}

// Sliders live updates
$('confSlider').addEventListener('input', e => $('confVal').textContent = parseFloat(e.target.value).toFixed(2));
$('overlapSlider').addEventListener('input', e => $('overlapVal').textContent = parseFloat(e.target.value).toFixed(2));
$('nmsSlider').addEventListener('input', e => $('nmsVal').textContent = parseFloat(e.target.value).toFixed(2));

// Upload Zone
$('dropZone').addEventListener('click', () => $('fileInput').click());
$('dropZone').addEventListener('dragover', e => { e.preventDefault(); $('dropZone').classList.add('dragover'); });
$('dropZone').addEventListener('dragleave', () => $('dropZone').classList.remove('dragover'));
$('dropZone').addEventListener('drop', e => {
  e.preventDefault();
  $('dropZone').classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    handleFileSelected(e.dataTransfer.files[0]);
  }
});
$('fileInput').addEventListener('change', e => {
  if (e.target.files.length) handleFileSelected(e.target.files[0]);
});

function handleFileSelected(file) {
  currentFile = file;
  currentSampleName = null;
  document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
  $('dropZone').querySelector('.drop-title').textContent = file.name;
  $('dropZone').querySelector('.drop-sub').textContent = (file.size / 1024).toFixed(1) + ' KB';
  runAnalysis();
}

// Run Analysis
async function runAnalysis() {
  if (!currentFile && !currentSampleName) return;

  $('perfText').textContent = 'Analyzing...';
  const conf = $('confSlider').value;
  const overlap = $('overlapSlider').value;
  const nms = $('nmsSlider').value;
  const topHigh = $('chkTopHighlight').checked;
  const colColors = $('chkColColors').checked;
  const banner = $('chkBanner').checked;

  try {
    let res;
    if (currentFile) {
      const fd = new FormData();
      fd.append('image', currentFile);
      const url = `/api/analyze-image?confidence=${conf}&overlap_threshold=${overlap}&nms_iou=${nms}&show_top_highlight=${topHigh}&show_column_colors=${colColors}&show_formula_banner=${banner}`;
      res = await fetch(url, { method: 'POST', body: fd });
    } else {
      const url = `/api/analyze-sample?filename=${encodeURIComponent(currentSampleName)}&confidence=${conf}&overlap_threshold=${overlap}&nms_iou=${nms}&show_top_highlight=${topHigh}&show_column_colors=${colColors}&show_formula_banner=${banner}`;
      res = await fetch(url, { method: 'POST' });
    }

    if (!res.ok) {
      const err = await res.json();
      alert('Analysis Error: ' + (err.detail || 'Failed'));
      $('perfText').textContent = 'Error';
      return;
    }

    const data = await res.json();
    lastAnalysisResult = data;
    renderResults(data);
  } catch (err) {
    alert('Network/Server error: ' + err.message);
    $('perfText').textContent = 'Error';
  }
}

function renderResults(data) {
  $('statRows').textContent = data.total_rows || 0;
  $('statTop').textContent = data.top_row_cartons || 0;
  $('statEst').textContent = data.estimated_total_cartons || 0;
  $('statDet').textContent = data.total_cartons_detected || 0;

  $('formulaDisplay').innerHTML = `Total Estimated = <span class="formula-highlight">${data.top_row_cartons}</span> (Top) × <span class="formula-highlight">${data.total_rows}</span> (Rows) = <span class="formula-highlight">${data.estimated_total_cartons} Cartons</span>`;
  $('columnSummaryText').textContent = `Columns Detected: ${data.columns_count} | Dim: ${data.image_dimensions.width}×${data.image_dimensions.height}`;

  $('perfText').textContent = `${data.inference_time_ms} ms`;

  // Display Image
  $('emptyState').style.display = 'none';
  $('mainImg').style.display = 'block';
  updateImageView();

  $('btnDownload').disabled = false;
  $('btnExportJson').disabled = false;

  // Render Column breakdown
  const bGrid = $('breakdownGrid');
  bGrid.innerHTML = '';
  if (data.columns && data.columns.length > 0) {
    $('colDetailCount').textContent = `${data.columns.length} Vertical Stacks`;
    data.columns.forEach(col => {
      const pill = document.createElement('div');
      pill.className = 'col-stat-pill';
      pill.innerHTML = `
        <div class="col-pill-title">Column ${col.column_index}</div>
        <div class="col-pill-num">${col.cartons_count} <span style="font-size:0.75rem; font-weight:500; color:var(--text-muted)">layers</span></div>
      `;
      bGrid.appendChild(pill);
    });
  } else {
    bGrid.innerHTML = '<div style="color:var(--text-dim); font-size:0.85rem;">No vertical columns detected.</div>';
  }
}

function updateImageView() {
  if (!lastAnalysisResult) return;
  if (currentViewMode === 'annotated') {
    $('mainImg').src = lastAnalysisResult.annotated_image;
  } else {
    $('mainImg').src = lastAnalysisResult.original_image;
  }
}

// View Tabs
$('tabAnnotated').addEventListener('click', () => {
  $('tabAnnotated').classList.add('active');
  $('tabOriginal').classList.remove('active');
  currentViewMode = 'annotated';
  updateImageView();
});
$('tabOriginal').addEventListener('click', () => {
  $('tabOriginal').classList.add('active');
  $('tabAnnotated').classList.remove('active');
  currentViewMode = 'original';
  updateImageView();
});

$('btnAnalyze').addEventListener('click', () => runAnalysis());
$('btnReset').addEventListener('click', () => {
  currentFile = null;
  currentSampleName = null;
  lastAnalysisResult = null;
  document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
  $('dropZone').querySelector('.drop-title').textContent = 'Drop pallet image here';
  $('dropZone').querySelector('.drop-sub').textContent = 'or click to browse from device (JPG, PNG)';
  $('statRows').textContent = '0';
  $('statTop').textContent = '0';
  $('statEst').textContent = '0';
  $('statDet').textContent = '0';
  $('formulaDisplay').textContent = 'Total Estimated = Top Cartons (0) × Total Rows (0) = 0 Cartons';
  $('mainImg').style.display = 'none';
  $('emptyState').style.display = 'block';
  $('breakdownGrid').innerHTML = '<div style="color:var(--text-dim); font-size:0.85rem;">No breakdown data available yet.</div>';
  $('btnDownload').disabled = true;
  $('btnExportJson').disabled = true;
  $('perfText').textContent = 'Ready';
});

// Download
$('btnDownload').addEventListener('click', () => {
  if (!lastAnalysisResult) return;
  const a = document.createElement('a');
  a.href = lastAnalysisResult.annotated_image;
  a.download = `pallet_count_${lastAnalysisResult.filename || 'result'}.jpg`;
  a.click();
});

// Export JSON
$('btnExportJson').addEventListener('click', () => {
  if (!lastAnalysisResult) return;
  const jsonStr = JSON.stringify(lastAnalysisResult, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `pallet_count_report_${Date.now()}.json`;
  a.click();
});

// Camera Snapshot
$('btnCameraModal').addEventListener('click', async () => {
  $('camModal').classList.add('open');
  try {
    streamObj = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    $('camVideo').srcObject = streamObj;
  } catch (e) {
    alert('Camera access denied or unavailable: ' + e.message);
    closeCamera();
  }
});

function closeCamera() {
  $('camModal').classList.remove('open');
  if (streamObj) {
    streamObj.getTracks().forEach(t => t.stop());
    streamObj = null;
  }
}
$('closeCamModal').addEventListener('click', closeCamera);
$('btnCancelCam').addEventListener('click', closeCamera);

$('btnSnapPhoto').addEventListener('click', () => {
  const video = $('camVideo');
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(blob => {
    const file = new File([blob], `camera_snap_${Date.now()}.jpg`, { type: 'image/jpeg' });
    closeCamera();
    handleFileSelected(file);
  }, 'image/jpeg', 0.92);
});

initApp();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    print(f"Starting PalletVision Angled Carton Counter on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
