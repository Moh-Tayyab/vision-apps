# CONTEXT.md — Project Context

## Purpose
Warehouse pallet inspection system that counts cartons from an overhead camera and tracks worker picks in real-time.

## Users
- **Warehouse Operator**: Primary user who monitors carton counts during picking operations
- **Supervisor**: Views live feed and pick events for oversight
- **System Admin**: Configures camera and system settings

## Environment
- Warehouse with overhead camera mounted above pallet
- Network connectivity for Roboflow API calls
- Local machine running the FastAPI server
- Browser-based dashboard access

## Current State
- Working prototype with all core features implemented
- Detection, tracking, hand detection, and state machine functional
- REST API and WebSocket endpoints operational
- Dashboard UI with live video feed
- Frame Zero: Auto-count from first frame
- Static Reference: Fixed baseline until layer clears

## Related Systems
- Local YOLO model (.pt file) - custom trained carton detection
- Camera sources (USB, IP webcam, RTSP, mobile)
- No integration with warehouse management systems
