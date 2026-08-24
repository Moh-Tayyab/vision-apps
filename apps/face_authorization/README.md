# App 3 — Face Authorization

FastAPI service using Python **deepface**: enroll authorized persons' face
embeddings, then flag faces from live video as authorized/unauthorized.
API-only.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8003   # from this directory
# or
docker build -t face-auth . && docker run -p 8003:8003 -v face-data:/app/data face-auth
```

First enrollment/verification call downloads Facenet weights (needs internet,
cached in `/app/.deepface`). Embeddings persist in `/app/data/embeddings.json`.

Model: Facenet embeddings, OpenCV face detector, cosine distance threshold 0.40.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health + enrolled-persons count |
| GET | `/model/info` | Model config + enrolled persons |
| POST | `/persons/enroll` | Fields: `name`, `files` (1+ face images) |
| GET | `/persons` | List enrolled persons |
| DELETE | `/persons/{name}` | Remove a person's embeddings |
| POST | `/verify` | Image → per-face `authorized`/`unauthorized` |
| POST | `/ingest/frame` | Push one JPEG frame (mobile camera) |
| POST | `/ingest/frame/check` | Verify latest pushed frame; logs events |
| GET | `/events` | Recent authorization events |
| GET | `/stream` / `/stream/detect` | Raw / annotated live MJPEG |
