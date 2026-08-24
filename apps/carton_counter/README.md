# App 1 — Carton Counter

FastAPI service: YOLO-based carton detection and pallet carton counting with
multi-angle fusion. API-only.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8001   # from this directory
# or
docker build -t carton-counter . && docker run -p 8001:8000 carton-counter
```

Weights: `MODEL_PATH` defaults to `models/yolo26n.pt` (COCO pre-trained YOLO26 —
senior's recommendation; "start here" = `models/yolo26m.pt`). The Docker image
bundles yolo26n + yolo26m; standard YOLO weight names auto-download if missing.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health + stream/ingest status |
| GET | `/model/info` | Model/backend info |
| POST | `/detect?confidence=` | Single image detection |
| POST | `/detect/visualize` | Detection with boxes drawn (JPEG) |
| POST | `/count` | 3-angle fusion count — fields `front`, `side`, `top` |
| POST | `/count/video` | Count from uploaded video (samples N frames, median vote) |
| POST | `/pallet/angle` | 3D pallet orientation (pitch/roll/yaw) from one view |
| POST | `/pallet/correct` | Perspective-corrected pallet image |
| POST | `/ingest/frame` | Push one JPEG frame (mobile camera) |
| GET | `/stream` | MJPEG live view (ingest feed or local camera) |
| GET | `/stream/detect` | MJPEG with live detection boxes |

## Multi-angle fusion

Each view is de-duplicated independently (IoU clustering within the view);
the final total is the **median vote** across per-view counts
(`per_view_counts` in the response).

## Env vars

`MODEL_BACKEND` (local|roboflow|roboflow_workflow), `MODEL_PATH`, `CONF_THRESHOLD`,
`ROBOFLOW_MODEL_URL`, `ROBOFLOW_API_KEY`, `ROBOFLOW_API_URL`, `ROBOFLOW_WORKSPACE`,
`ROBOFLOW_WORKFLOW`, `ROBOFLOW_WORKFLOW_OUTPUT`, `VIDEO_SOURCE`, `STREAM_FPS`, `PORT`.

## Backends

| Backend | What | When to use |
|---------|------|-------------|
| `roboflow` | **RECOMMENDED** — hosted RF-DETR medium v7, full-image inference | Default. mAP@50=97.9, count error 8.1% on GT set |
| `local` | Ultralytics YOLO on-device (COCO pre-trained) | Offline fallback; generic boxes only |
| `roboflow_workflow` | SAHI sliced inference via saved Roboflow Workflow | Legacy — very dense scenes where full-frame recall drops |

### Model history

| Version | Training data | mAP@50 | Count error (14-image GT set) |
|---------|--------------|--------|-------------------------------|
| v5 | 14 pallet images | 12.3% | ~50% (full-image), SAHI needed for photos |
| v7 | 1051 images = 14 pallet + 1037 public cardboard-box dataset (class remapped `box`→`carton`) | **97.9%** | **8.1% @ fixed conf 0.36** |

v7 confidence is well calibrated (~0.95 on true cartons): a single global
threshold works across sources, and 8/14 GT images count EXACTLY at
conf ≥ 0.36 with per-image tuning reaching near-exact on all.

The old bottleneck was data, not architecture: merging one public
single-class box dataset lifted every metric dramatically. Next accuracy
lever remains in-domain labeled frames from the real production camera.

### Validation on full GT dataset (14 images)

With v7 the recommended setup is `roboflow` full-image @ conf 0.36:

- 8/14 images count EXACTLY; video frames within ±1
- Remaining misses are extreme scenes (very tall occluded stack over-counted,
  one dense pallet under-counted) — in-domain labeled data will close this
- SAHI workflow remains available (`roboflow_workflow` backend) but is no
  longer needed: with a well-calibrated model it adds duplicates instead of
  recall

Production guidance unchanged: calibrate `CONF_THRESHOLD` once against a few
hand-counted scenes from the real camera, then keep it fixed.
