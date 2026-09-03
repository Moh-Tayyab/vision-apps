# CONTRACT.md — API Contract

## Base URL
```
http://localhost:8001
```

## REST Endpoints

### GET /health
**Purpose**: System health check
**Response**:
```json
{
  "status": "healthy" | "degraded",
  "detector_configured": true | false,
  "mode": "live_tracking"
}
```

### GET /model/info
**Purpose**: Detection model configuration
**Response**:
```json
{
  "backend": "local_yolo",
  "model_path": "best.pt",
  "confidence": 0.36,
  "nms_iou": 0.35,
  "ios_thresh": 0.58,
  "last_inference_ms": 45.2
}
```

### GET /live/info
**Purpose**: Current live counter status
**Response**:
```json
{
  "mode": "live_tracking",
  "initial_cartons": 24,
  "tracked_cartons": 12,
  "picked_cartons": 8,
  "auto_count_done": true
}
```

### POST /live/init
**Purpose**: Initialize live counter with ROI settings
**Query Parameters**:
- `top_row_only` (bool, default: true): Only track topmost row
- `roi_x1`, `roi_y1`, `roi_x2`, `roi_y2` (int): ROI coordinates

**Behavior**:
- System auto-detects initial cartons from first 5 stable frames
- No manual initial count needed

**Response**:
```json
{
  "status": "initialized",
  "top_row_only": true,
  "roi": {"x1": 0, "y1": 0, "x2": 1920, "y2": 1080}
}
```

### POST /live/frame
**Purpose**: Process single video frame
**Request**: Multipart form data
- `image` (file, required): Video frame to process
- `confidence` (float, 0.05-0.95, default: 0.36): Detection confidence
- `top_row_only` (bool, default: true): Filter mode
- `annotate` (bool, default: true): Include annotated frame

**Response**:
```json
{
  "total_active": 12,
  "top_row_count": 12,
  "total_picked": 8,
  "layer_counts": {"0": 12},
  "cartons_by_state": {"present": 12},
  "hand_detected": false,
  "picking_in_progress": false,
  "frame_time_ms": 45.2,
  "tracks": [...],
  "events": [...],
  "top_row_only": true,
  "annotated_frame": "data:image/jpeg;base64,..."
}
```

### POST /live/reset
**Purpose**: Reset live counter state
**Response**:
```json
{
  "status": "reset"
}
```

### GET /camera/status
**Purpose**: Camera connection status
**Response**:
```json
{
  "health": {
    "source_type": "http_mjpeg",
    "uri": "http://10.120.247.162:8080/video",
    "connected": true,
    "fps": 25.0,
    "frame_count": 1234,
    "resolution": "640x480",
    "medium": "Wi-Fi Network / RTSP"
  },
  "available_usb_cameras": [...]
}
```

### POST /camera/connect
**Purpose**: Connect to camera source
**Query Parameters**:
- `source_type` (string, required): "usb" | "http_mjpeg" | "rtsp" | "mobile"
- `uri` (string, required): Camera URI
- `fps` (int, default: 25): Target frame rate
- `width` (int, default: 640): Frame width
- `height` (int, default: 480): Frame height
- `rotation` (int, default: 0): Rotation in degrees (0, 90, 180, 270)

**Response**:
```json
{
  "status": "connected",
  "message": "Connecting",
  "uri": "http://10.120.247.162:8080/video"
}
```

### POST /camera/disconnect
**Purpose**: Disconnect camera
**Response**:
```json
{
  "status": "disconnected"
}
```

### POST /camera/ingest
**Purpose**: Ingest frame from mobile browser (push mode)
**Request**: Multipart form data
- `image` (file, required): Video frame

**Response**:
```json
{
  "status": "ok",
  "frame_count": 1234
}
```

### GET /camera/snapshot
**Purpose**: Get latest frame as JPEG
**Response**: Image/jpeg binary

### GET /
**Purpose**: Dashboard UI
**Response**: HTML page

## WebSocket Endpoints

### /ws/live
**Purpose**: Live video streaming with client-provided frames
**Protocol**: WebSocket
**Client sends**: Binary (JPEG frame)
**Server sends**:
- Binary: Annotated JPEG frame
- JSON: Count data

**JSON Message Format**:
```json
{
  "total_active": 12,
  "top_row_count": 12,
  "total_picked": 8,
  "hand_detected": false,
  "picking_in_progress": false,
  "events": [...],
  "top_row_only": true
}
```

### /ws/camera
**Purpose**: Live camera feed with real-time detection
**Protocol**: WebSocket
**Server sends**:
- Binary: Annotated JPEG frame (30 FPS)
- JSON: Full count data

**JSON Message Format**:
```json
{
  "current_layer": 1,
  "initial_row_cartons": 24,
  "cartons_remaining": 12,
  "cartons_removed_from_row": 12,
  "total_active": 12,
  "total_picked": 8,
  "hand_detected": false,
  "picking_in_progress": false,
  "events": [...],
  "frame_time_ms": 45.2,
  "top_row_only": true,
  "layer_transition_triggered": false
}
```

## Event Types

### Pick Event
```json
{
  "track_id": 5,
  "event": "picked",
  "time": 1693000000.123,
  "row": 0,
  "layer": 0
}
```

### Layer Cleared Event
```json
{
  "event": "layer_1_cleared",
  "layer": 1,
  "time": 1693000000.456,
  "track_id": 0
}
```

### Timeout Removal Event
```json
{
  "track_id": 3,
  "event": "removed_timeout",
  "time": 1693000000.789,
  "row": 0,
  "layer": 0
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Image is empty"
}
```

### 503 Service Unavailable
```json
{
  "detail": "No frame available"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Connection failed: Port 8080 unreachable"
}
```
