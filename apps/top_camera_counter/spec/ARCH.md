# ARCH.md — System Architecture

## Overview
Top Camera Carton Counter is a real-time warehouse pallet inspection system that uses computer vision to count cartons and track worker picks from an overhead camera.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser Dashboard                        │
│  (HTML/CSS/JS - Live Video, Stats, Event Log)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ WebSocket / HTTP
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server                            │
│  (main.py - REST API + WebSocket + Dashboard)                │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Camera Manager │ │  Live Counter   │ │   Dashboard     │
│  (camera_       │ │  (live_counter  │ │   (HTML in      │
│   manager.py)   │ │   .py)          │ │    main.py)     │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │
          │                   │
          ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│  Camera Source  │ │    Detector     │
│  (streamer.py)  │ │  (detector.py)  │
└─────────────────┘ └─────────────────┘
                          │
                          ▼
                  ┌─────────────────┐
                  │  Local YOLO     │
                  │  Model (.pt)    │
                  └─────────────────┘
```

## Components

### 1. Camera Manager (`camera_manager.py`)
- **Purpose**: Manages video acquisition from multiple camera sources
- **Responsibilities**:
  - Camera connection/disconnection
  - Background frame capture with auto-reconnect
  - Health monitoring (FPS, resolution, connection status)
  - Support for USB, HTTP MJPEG, RTSP, mobile push sources

### 2. Live Counter (`live_counter.py`)
- **Purpose**: Core counting logic with detection integration
- **Responsibilities**:
  - Frame processing pipeline
  - Top-row/layer spatial filtering
  - ROI-based detection filtering
  - Layer lifecycle management
  - Annotation overlay generation
  - Auto-count: Detect initial cartons from first stable frames
  - Static Reference: Fixed baseline until layer clears

### 3. Detector (`detector.py`)
- **Purpose**: YOLO-based carton detection
- **Responsibilities**:
  - Local YOLO model inference (.pt file)
  - NMS (Non-Maximum Suppression) deduplication
  - IoU and IOS (Intersection over Smaller) filtering
  - Plausibility checks for detected boxes

### 4. Tracker (`tracker.py`)
- **Purpose**: Multi-object tracking with unique IDs
- **Responsibilities**:
  - ByteTrack-inspired tracking algorithm
  - IoU-based detection-to-track matching
  - Coordinate smoothing for stable bounding boxes
  - Track lifecycle management

### 5. Worker Detector (`worker_detector.py`)
- **Purpose**: Detect worker hands for pick detection
- **Responsibilities**:
  - MediaPipe pose/hand detection
  - Hand position tracking
  - Hand velocity calculation
  - Proximity detection to cartons

### 6. State Machine (`state_machine.py`)
- **Purpose**: Carton lifecycle state management
- **States**:
  - `PRESENT`: Carton visible and tracked
  - `BEING_PICKED`: Hand detected near carton
  - `REMOVED`: Carton picked/removed
  - `OCCLUDED`: Temporarily hidden
- **Transitions**: Hand proximity triggers picks, timeout triggers removals

### 7. Streamer (`streamer.py`)
- **Purpose**: Frame buffer and transform utilities
- **Responsibilities**:
  - Thread-safe frame buffer
  - Image rotation/transform operations

## Data Flow

### Frame Processing Pipeline
```
Camera Frame → Detector (YOLO) → NMS Filter → ROI Filter
    → Top-Row Filter → Tracker → State Machine → Result
```

### WebSocket Data Flow
```
Client ← WebSocket ← Server
  ├── Binary: Annotated JPEG frame
  └── JSON: Count data + events
```

## Data Models

### LiveCountResult
```python
{
  current_layer: int,           # Active layer number
  initial_row_cartons: int,     # Cartons at layer start
  cartons_remaining: int,       # Currently visible
  cartons_removed_from_row: int, # Removed this layer
  total_active: int,            # Total active cartons
  total_picked: int,            # Total picked all layers
  top_row_count: int,           # Top row filter count
  layer_counts: Dict[int, int], # Count per layer
  cartons_by_state: Dict,       # State breakdown
  hand_detected: bool,          # Worker hand visible
  picking_in_progress: bool,    # Active pick happening
  frame_time_ms: float,         # Processing time
  tracks: List[dict],           # Active tracks
  events: List[dict],           # Pick events
  top_row_only: bool,           # Filter mode
  layer_transition_triggered: bool  # Layer cleared
}
```

### CartonTrack
```python
{
  track_id: int,                # Unique ID
  state: CartonState,           # Current state
  bbox: Tuple[float,float,float,float],  # Bounding box
  confidence: float,            # Detection confidence
  row: int,                     # Row position
  layer: int,                   # Layer position
  first_seen: float,            # Timestamp
  last_seen: float,             # Timestamp
  occluded_frames: int,         # Frames not detected
  picked_by_hand: bool          # Hand pick detected
}
```

## Error Handling
- Detection failures return empty results (no crash)
- Camera disconnect triggers auto-reconnect
- Network errors logged but don't stop processing
- Invalid frames skipped gracefully

## Performance Considerations
- Frame processing is CPU-bound (OpenCV + numpy)
- WebSocket sends frames at ~30 FPS with JPEG compression
- Background detection runs asynchronously in WebSocket mode
- Tracker uses simplified ByteTrack for speed
