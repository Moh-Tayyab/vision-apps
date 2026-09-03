# SPEC.md — Top Camera Carton Counter

## Project Overview
Real-time carton counting system for warehouse pallet inspection using overhead camera.

## Read Order
1. `spec/CONTEXT.md` - Project context and users
2. `spec/MEMORY.md` - Frozen decisions
3. `spec/CONSTRAINTS.md` - Hard rules
4. `spec/SECURITY.md` - Security considerations
5. `spec/ARCH.md` - System architecture
6. `spec/CONTRACT.md` - API contract
7. `spec/backend_specs/tasks/task_index.md` - Task breakdown

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Place your trained YOLO model
# Copy best.pt to project root, or set YOLO_MODEL_PATH in .env

# Run server
python main.py
```

## Architecture Summary
- **Backend**: FastAPI + WebSocket
- **Detection**: Local YOLO model (.pt file)
- **Tracking**: ByteTrack-inspired algorithm
- **Hand Detection**: MediaPipe
- **State Machine**: Carton lifecycle management

## API Endpoints
- `GET /health` - Health check
- `POST /live/init` - Initialize counter
- `POST /live/frame` - Process frame
- `POST /camera/connect` - Connect camera
- `WS /ws/camera` - Live stream + detection

## File Structure
```
├── main.py              # FastAPI app + dashboard
├── detector.py          # YOLO detection
├── tracker.py           # Object tracking
├── live_counter.py      # Counting logic
├── state_machine.py     # Carton lifecycle
├── camera_manager.py    # Camera sources
├── worker_detector.py   # Hand detection
├── streamer.py          # Frame buffer
├── requirements.txt     # Dependencies
├── Dockerfile           # Container setup
└── spec/                # This spec tree
```
