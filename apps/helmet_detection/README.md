# App 2 — Helmet Detection

FastAPI service: detect persons with/without helmets from images or live
mobile-camera frames. API-only.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8002   # from this directory
# or
docker build -t helmet-detection . && docker run -p 8002:8002 helmet-detection
```

## Model backends

- **local** (default): COCO `yolo26n.pt` detects *persons only* — every person
  gets `status: "unknown"`. Point `MODEL_PATH` at weights trained on
  person/helmet/head classes for real helmet/no-helmet verdicts.
- **roboflow**: any hosted model with `person`/`helmet`/`head` classes, e.g.
  `https://detect.roboflow.com/dataperson/safety-helmet-dataset-uvh1t/1`
  (set `MODEL_BACKEND=roboflow` + `ROBOFLOW_MODEL_URL` + `ROBOFLOW_API_KEY`).

Status logic: a `helmet` box inside the top 70% of a person box → `helmet`;
a bare-`head` box → `no_helmet`; head-only datasets also supported.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health + ingest status |
| GET | `/model/info` | Backend + class config |
| POST | `/detect?confidence=` | Image → per-person helmet status + violations |
| POST | `/detect/visualize` | Annotated JPEG (green/red/orange) |
| POST | `/ingest/frame` | Push one JPEG frame (mobile camera) |
| POST | `/ingest/frame/check` | Detect on latest pushed frame; logs violations |
| GET | `/violations` | Recent no-helmet events |
| GET | `/stream/detect` | Live annotated MJPEG |
