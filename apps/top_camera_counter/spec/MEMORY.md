# MEMORY.md — Project Decisions

## dec-detection-backend
- **Decision**: Use local YOLO model (.pt file) for carton detection
- **Rationale**: No API dependency, faster inference, custom trained model for specific use case
- **Trade-off**: Requires model training, local GPU recommended for speed
- **Status**: Frozen

## dec-tracking-algorithm
- **Decision**: ByteTrack-inspired multi-object tracker
- **Rationale**: Good balance of accuracy and speed for warehouse pallet counting, handles occlusion well
- **Trade-off**: Simplified version of full ByteTrack, may miss some edge cases
- **Status**: Frozen

## dec-hand-detection
- **Decision**: MediaPipe for worker hand/pose detection
- **Rationale**: Free, runs locally, good hand tracking capabilities
- **Trade-off**: Additional dependency, may need tuning for warehouse lighting
- **Status**: Frozen

## dec-api-framework
- **Decision**: FastAPI with WebSocket support
- **Rationale**: Modern async framework, auto-generated docs, WebSocket for real-time streaming
- **Trade-off**: Python GIL limits true parallelism, but sufficient for this use case
- **Status**: Frozen

## dec-frontend-approach
- **Decision**: Embedded HTML/CSS/JS in main.py
- **Rationale**: Single-file deployment, no build step required, simple for warehouse use
- **Trade-off**: Not maintainable for large UIs, but sufficient for dashboard
- **Status**: Frozen

## dec-state-machine
- **Decision**: PRESENT -> BEING_PICKED -> REMOVED -> OCCLUDED lifecycle
- **Rationale**: Captures all carton states needed for accurate counting
- **Trade-off**: State transitions need tuning for edge cases
- **Status**: Frozen

## dec-layer-transition
- **Decision**: Auto-transition when visible cartons drop to 0 for 6 frames
- **Rationale**: Prevents false transitions from temporary detection failures
- **Trade-off**: 6-frame delay before transition
- **Status**: Frozen

## dec-roi-filtering
- **Decision**: Configurable PalletROI for spatial filtering
- **Rationale**: Allows focusing on specific pallet area, reduces false detections
- **Trade-off**: Requires manual ROI configuration per camera setup
- **Status**: Frozen

## dec-auto-count
- **Decision**: Auto-detect initial cartons from first stable frames
- **Rationale**: No manual input needed, system learns baseline automatically
- **Trade-off**: Needs 5 stable frames before locking baseline
- **Status**: Frozen

## dec-static-reference
- **Decision**: initial_row_cartons stays fixed once set until layer clears
- **Rationale**: Consistent baseline for removal calculation (Removed = Initial - Current)
- **Trade-off**: No mid-layer adjustment if initial count was wrong
- **Status**: Frozen
