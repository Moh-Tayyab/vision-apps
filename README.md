# Vision Apps (3 Independent FastAPI Services)

Three independent, API-only apps (no UI / no Streamlit), each with its own
FastAPI app and Dockerfile:

| App | Directory | Port | Purpose |
|-----|-----------|------|---------|
| 1 | `apps/carton_counter/` | 8001 | Carton counting on pallets (multi-angle fusion) |
| 2 | `apps/helmet_detection/` | 8002 | Helmet vs no-helmet detection on live camera |
| 3 | `apps/face_authorization/` | 8003 | Authorized/unauthorized person via deepface embeddings |

## Quick Start (all apps)

```bash
docker compose up --build
```

Each app is fully independent — you can also build/run any one alone:

```bash
docker build -t carton-counter ./apps/carton_counter
docker run -p 8001:8000 carton-counter
```

## Live Camera Pattern (shared by all 3 apps)

Apps consume mobile-camera video by frame push (works from any device,
no WebRTC needed):

```bash
# From the phone/client, push JPEG frames repeatedly:
curl -X POST http://localhost:8001/ingest/frame -F "file=@frame.jpg"

# Then view or consume the live stream:
#   GET /stream          raw MJPEG
#   GET /stream/detect   annotated MJPEG (App 1 & 2)
```

## Roboflow Cloud Backends

Set in `.env` (gitignored):

- `ROBOFLOW_API_KEY=rf_...`
- `ROBOFLOW_MODEL_URL=https://detect.roboflow.com/<workspace>/<project>/<version>`

Then set `MODEL_BACKEND=roboflow` for that service.

## Tests

```bash
python tests/test_detector.py            # App 1 smoke test (server must be running)
CARTON_COUNTER_URL=http://localhost:8010 python tests/test_detector.py
```
