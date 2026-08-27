# Carton Counter Project - Context & Plan

## Project Overview
Count cartons on pallet using YOLO detection with multi-angle fusion.

## Senior Requirements (Muhammad Usama)
- 3 independent apps: Carton Counter, Helmet Detection, Face Authorization
- Start with Carton Counter (App 1)
- Use COCO pre-trained YOLO (no training needed for MVP)
- In production, cameras will be set up by us
- Cartons can be of DIFFERENT sizes on a pallet
- Docker Compose deployment
- FastAPI backend (no Streamlit UI)
- WebRTC / Browser Stream for live camera

## Architecture Decisions

### Carton Counting Strategy - Multi-View Fusion (BEST for Production)
```
Camera 1 (Front)  → YOLO Detect → Detections_1 ─┐
Camera 2 (Side)   → YOLO Detect → Detections_2 ─┼→ Fusion Engine → Total Count
Camera 3 (Top)    → YOLO Detect → Detections_3 ─┘
```

### Why Multi-View Fusion?
1. Eliminates blind spots (single camera misses stacked cartons)
2. Handles different carton sizes naturally
3. Top camera sees layers/stacking
4. Production-ready (cameras fixed, calibration done once)

### Fusion Algorithm
- IoU-based clustering (same carton from different angles = high IoU)
- Threshold: 0.3 (configurable)
- Mode/median voting for final count

## API Endpoints (App 1 - Carton Counter)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/detect` | Single image detection |
| POST | `/count` | 3-angle merged count |
| POST | `/count/video` | Video processing |
| POST | `/detect/visualize` | Detection with visualization |

## Tech Stack
- Python 3.12
- FastAPI + Uvicorn
- Ultralytics YOLOv11 (COCO pre-trained)
- OpenCV
- Docker

## Files Structure
- `apps/carton_counter/main.py` - FastAPI app
- `apps/carton_counter/detector.py` - YOLO detection engine
- `apps/carton_counter/counter.py` - Counting logic (single + multi-angle)
- `apps/carton_counter/models/yolo11n.pt` - COCO pre-trained model
- `apps/carton_counter/requirements.txt` - Dependencies
- `apps/carton_counter/Dockerfile` - Docker image

## Implementation Status
- [x] Project scaffolding
- [x] detector.py - YOLO engine
- [x] counter.py - counting logic
- [x] main.py - FastAPI endpoints
- [x] requirements.txt
- [x] Dockerfile
- [ ] Test with actual carton images
- [ ] Deploy & test endpoints

## App 3 - Face Authorization Status
- [x] face_engine.py - deepface/Facenet embedding store (JSON-backed)
- [x] main.py - FastAPI endpoints (enroll/verify/ingest/stream)
- [x] streamer.py - MJPEG frame buffer
- [x] requirements.txt (fixed: tf-keras added, version pins relaxed)
- [x] Dockerfile (volumes for embeddings + model cache)
- [x] DETECTOR_BACKEND fixed: "opencv" → "retinaface" (OpenCV 5.x removed cascade XMLs)
- [x] Full FaceEngine test: enroll → identify → distance=0.0 → authorized=True → remove ✅
- [x] face_engine.py - Photo storage: saves front-facing photo to `data/photos/{name}/photo.jpg`
- [x] main.py - New endpoint: `GET /persons/{name}/photo` (serves enrollment photo)
- [x] streamlit_app.py - Streamlit admin dashboard (4 pages: Dashboard, Enroll, Manage, Live)
- [x] Dockerfile.streamlit - Separate lightweight Docker image for Streamlit
- [x] requirements-streamlit.txt - Streamlit dependencies (no deepface needed)
- [x] docker-compose.yml - Added `face-auth-ui` service (port 8501)

## Roboflow Project
- Project: `muhammad-tayyab-iqnwv/carton-counter-demo`
- Type: Object Detection
- Class: `carton` (64 annotations)
- 9 train images (3 annotated: 13, 19, 32 cartons)
- API Key in .env

## Next Steps
1. Test the API with sample images
2. Fine-tune on carton dataset if needed
3. Add video processing
4. Build App 2 (Helmet Detection)
5. ~~Build App 3 (Face Authorization)~~ ✅ Done (with Streamlit admin UI)
6. Docker Compose for all 3 apps

## Senior's Key Points
- "Try to work separately - one app at a time"
- "For carton detection, assume cameras will be set up by us in production"
- "Cartons can be of different sizes on a pellet"
