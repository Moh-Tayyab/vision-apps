# App 3 — Face Authorization

FastAPI service using Python **deepface**: enroll authorized persons' face
embeddings, then flag faces from live video as authorized/unauthorized.
Includes a **Streamlit admin dashboard** for easy user management.

## Run

### FastAPI Backend (port 8003)
```bash
uvicorn main:app --host 0.0.0.0 --port 8003   # from this directory
# or
docker build -t face-auth . && docker run -p 8003:8003 -v face-data:/app/data face-auth
```

### Streamlit Admin Dashboard (port 8501)
```bash
streamlit run streamlit_app.py --server.port 8501   # from this directory
# or
docker build -t face-auth-ui -f Dockerfile.streamlit . && docker run -p 8501:8501 -e FASTAPI_URL=http://localhost:8003 face-auth-ui
```

### Docker Compose (both services)
```bash
docker compose up --build   # from project root
# Backend: http://localhost:8003
# Admin UI: http://localhost:8501
```

## Streamlit Dashboard Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Stats (persons, embeddings, events), enrolled persons list, recent events table |
| **Enroll User** | Upload 1 clear front-facing photo + name → one-click enrollment |
| **Manage Users** | View all enrolled users with photos, delete button per user |
| **Live Detection** | Upload image to verify, or embed live camera stream from FastAPI |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health + enrolled-persons count |
| GET | `/model/info` | Model config + enrolled persons |
| POST | `/persons/enroll` | Fields: `name`, `files` (1+ face images) |
| GET | `/persons` | List enrolled persons |
| GET | `/persons/{name}/photo` | Serve enrollment photo (JPEG) |
| DELETE | `/persons/{name}` | Remove a person's embeddings + photo |
| POST | `/verify` | Image → per-face `authorized`/`unauthorized` |
| POST | `/ingest/frame` | Push one JPEG frame (mobile camera) |
| POST | `/ingest/frame/check` | Verify latest pushed frame; logs events |
| GET | `/events` | Recent authorization events |
| GET | `/stream` / `/stream/detect` | Raw / annotated live MJPEG |

## Photo Storage

Each enrolled person gets a directory under `data/photos/<name>/` containing
the front-facing enrollment photo. MVP stores 1 photo; the structure is ready
for 3-4 photos per employee in a future update.
