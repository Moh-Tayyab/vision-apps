# CONSTRAINTS.md — Hard Rules

## sec-model-file
- YOLO model file (.pt) must be present in project root or path specified in YOLO_MODEL_PATH
- Model file must not be committed to version control (large file)

## sec-network
- System must handle model loading failures gracefully
- Detection errors must not crash the application

## perf-frame-rate
- WebSocket video stream must maintain minimum 15 FPS for usability
- Frame processing must not block video delivery

## rel-state-persistence
- In-memory state only - no database required
- System state resets on server restart (by design)

## compat-camera
- Must support at least: USB webcams, HTTP MJPEG streams, RTSP
- Camera connection failures must be reported clearly

## ui-dashboard
- Dashboard must work in modern browsers (Chrome, Firefox, Edge)
- No external CDN dependencies (all assets self-contained)

## api-backward-compatibility
- REST API endpoints must not change without version bump
- WebSocket message format must remain backward compatible
