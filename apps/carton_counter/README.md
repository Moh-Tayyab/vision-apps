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
| `local` | Ultralytics YOLO on-device (COCO pre-trained) | No API key / offline; generic box detection |
| `roboflow` | Hosted fine-tuned model, full-image inference | Simple cloud setup (`carton-counter-demo/5`) |
| `roboflow_workflow` | **SAHI sliced inference** via saved Roboflow Workflow: Image Slicer (640px tiles, 25% overlap) → `carton-counter-demo/5` → NMS stitch | **Best accuracy** — recommended |

Why slicing: cartons occupy few pixels after full-frame resize, so the model
misses most of them. With 640px tiles + stitch, counts on real pallet photos
improved dramatically (ground truth from hand-labeled dataset):

| Image | GT | Full-image best | SAHI @ per-image optimal threshold |
|-------|----|-----------------|-------------------------------------|
| pallet1 | 27 | 16 (59%) | **27 — exact** (@0.10) |
| img2 | 32 | 20 (63%) | 36 or 32±4 (@0.10) |
| img3 | 18 | 10 (56%) | 22 (@0.11) |
| img4 | 12 | 4 (33%) | 12 — exact (@0.14) |

The workflow returns every detection ≥ 0.01; `CONF_THRESHOLD` filters
client-side (no re-inference when tuning). Global default `0.11`; per-scene
tuning recovers near-exact counts. Confidence calibration improves as more
labeled frames are added to the training set.

Setup: `MODEL_BACKEND=roboflow_workflow ROBOFLOW_API_KEY=... CONF_THRESHOLD=0.11`
(workspace/workflow already default to `muhammad-tayyab-iqnwv/carton-counter-sahi`).
