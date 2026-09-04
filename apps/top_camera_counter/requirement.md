# Top Camera Carton Counter - Warehouse Pallet Inspector

## Goal
Real-time carton counting system using an overhead/top camera that tracks cartons on a pallet, detects worker hand picks, and calculates how many cartons have been removed from each layer. The system auto-transitions to the next layer when the current layer is cleared.

## User Journeys
1. **Warehouse Operator**: Connects overhead camera, sets initial carton count per layer, starts live feed, monitors real-time count on dashboard
2. **Supervisor**: Views live annotated video feed with HUD overlay showing layer status, picks, and events
3. **System Admin**: Configures camera source (USB, IP webcam, RTSP, mobile), sets confidence thresholds, manages ROI

## Features
- **Carton Detection**: YOLO-based detection via Roboflow cloud API with NMS deduplication
- **Object Tracking**: ByteTrack-inspired multi-object tracker with unique IDs and coordinate smoothing
- **Hand/Pick Detection**: MediaPipe-based worker hand detection to identify picking events
- **Layer Lifecycle**: Auto-detects when a layer is cleared and transitions to next layer
- **Top Row Filtering**: Option to only track and count the topmost visible row of cartons
- **State Machine**: Tracks each carton through PRESENT -> BEING_PICKED -> REMOVED -> OCCLUDED states
- **Camera Management**: Supports USB webcams, HTTP MJPEG streams, RTSP, and mobile browser push
- **Real-time Dashboard**: HTML/JS dashboard with live video, stats cards, event log
- **WebSocket Streaming**: Real-time video + JSON data via WebSocket
- **REST API**: Endpoints for health, model info, live frame processing, camera control

## Tech Stack
- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Detection**: Roboflow YOLO API (cloud inference)
- **Tracking**: Custom ByteTrack implementation (numpy-based)
- **Hand Detection**: MediaPipe (pose/hand detection)
- **Computer Vision**: OpenCV (image processing, annotation, video capture)
- **Communication**: WebSocket + REST API
- **Frontend**: Vanilla HTML/CSS/JS (embedded in main.py)

## Design
- Dark theme dashboard (background: #0b0f19, cards: #151d30)
- Primary color: Sky blue (#38bdf8)
- Status colors: Emerald (#10b981) for active, Amber (#f59e0b) for warnings, Red (#ef4444) for alerts
- Font: System font stack (-apple-system, BlinkMacSystemFont, Segoe UI, Roboto)
- HUD overlay on video: Dark banner with real-time stats

## Non-Goals
- No user authentication or multi-user support
- No persistent database storage (state is in-memory only)
- No mobile app (dashboard is browser-based)
- No training of custom models (uses pre-trained Roboflow model)
- No video recording or playback

## Constraints
- ROBOFLOW_API_KEY must be set in environment or .env file
- Node 22+ required for SLC tooling (not for this Python project)
- Camera source must be reachable before connecting
- WebSocket connections are single-client per stream
- Frame processing is synchronous (blocking) in API mode, async in WebSocket mode
