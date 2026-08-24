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

- **roboflow** (default): our hosted RF-DETR helmet model
  `safety-helmet-dataset-uvh1t-aavk1/1` (head/helmet/person), trained on a
  1088-image public safety-helmet dataset — mAP@50=95.5, recall=96.4.
  Validated end-to-end on an online construction video: stable per-frame
  person counts, correct helmet verdicts, violations flagged.
- **local**: COCO `yolo26n.pt` detects *persons only* — every person gets
  `status: "unknown"`. Point `MODEL_PATH` at weights trained on
  person/helmet/head classes for offline use.

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
