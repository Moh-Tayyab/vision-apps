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

`MODEL_BACKEND` (local|roboflow), `MODEL_PATH`, `CONF_THRESHOLD`,
`ROBOFLOW_MODEL_URL`, `ROBOFLOW_API_KEY`, `VIDEO_SOURCE`, `STREAM_FPS`, `PORT`.
