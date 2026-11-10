# 🚚 Three-Camera Truck Loading Carton Counter

A Flask application for counting cartons crossing a truck-loading line, using the canonical carton segmentation model and a synchronized three-video processing path.

---

## 🎯 Features

- **Directional Tripwire Counting**:
  - **Left-to-Right (+1 / Loaded)**: When worker/carton moves towards the truck.
  - **Right-to-Left (-1 / Returned)**: When worker/carton moves back from the truck.
- **Robust Multi-Object Tracking**: Utilizes **ByteTrack** for persistent ID association across frames and occlusions.
- **Hysteresis & Anti-Double Counting**: Built-in spatial hysteresis zone and frame cooldown preventing noisy double counting at the line boundary.
- **Central Multi-Camera Counting**: Recorded top, side, and flank videos are processed as one activity. Camera observations are fused into crossing events rather than summing camera-local totals.
- **Recorded-Video Association**: Two or three selected views can be processed together. When full physical calibration is unavailable, recorded mode uses synchronized tripwire-time matches; it assumes aligned videos and that each red line marks the same physical boundary. The repository has no experimentally validated association-window value; the legacy 1.2-second candidate interval is not validated evidence. Treat time-based results as provisional until matched events are manually annotated. Competing or single-camera observations are excluded from the count and shown as diagnostic evidence.
- **Conservative Central Count**: Loaded / Returned / Net include only unique crossing events associated across at least two selected camera views. The dashboard does not present a review queue or unconfirmed observations.
- **Optional Spatial Calibration**: A valid dock-plane calibration enables calibrated spatial association, but is not required for recorded-video pixel-tripwire counting. When calibration is absent, the fallback matches synchronized crossing events by time and direction; it does not compare raw pixels between cameras. The checked-in tripwire defaults are starting positions, not independently verified physical boundaries.
- **Calibration Evidence**: A homography is accepted for counting only when it has surveyed references on the common loading-gate plane, at least four independent measured validation points, a measured reprojection error, and the same image resolution used by the video. Counters are in-memory and reset when a processing session restarts.
- **Live Camera Preview**: The live-camera panel can capture and preview three RTSP/HTTP streams or numeric USB camera indices, with per-camera connection health and reconnect attempts. Live capture is preview-only. A live three-camera inference/counting loop is not implemented, and calibration must not be treated as sufficient by itself to enable live counting.
- **Interactive Tripwire Placement**:
  - Click and drag the virtual vertical line directly on the screen or dashboard.
  - Keyboard shortcuts for fine adjustments.
- **Recorded Inputs**: Supports synchronized MP4 video processing from the three camera views.

---

## 📁 Project Structure

```text
apps/truck_loading_detection/
├── app.py                      # Interactive Flask web application & live streaming backend
├── three_camera_system/        # Synchronized multi-camera detection and crossing pipeline
├── visualizer.py               # HUD overlay, glow line, trails, and detection renderer
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerization specification
├── templates/
│   └── index.html              # Modern Web Dashboard UI
├── sample_truck_loading.mp4    # Sample warehouse loading video
├── best.pt                     # Canonical Production Carton Specialist Model (Instance Seg)
├── best_pt_v1_baseline.pt      # Rollback Baseline Model Checkpoint
└── README.md                   # Documentation
```

---

## 🧠 Model Architecture & Production Checkpoints

The active carton detector is fixed to the canonical checkpoint. Startup rejects another model path, a hash mismatch, a non-segmentation task, or a class map other than `{0: 'carton'}`.

| Contract | Expected value |
| :--- | :--- |
| Checkpoint | `apps/truck_loading_detection/best.pt` |
| SHA-256 | `98c6164c83905feb9db6d9bee6f48ab239c5d68a00efbe0484dbf3448333ad33` |
| Task | Instance segmentation |
| Class map | `{0: 'carton'}` |
| Default camera crossing signs | `cam_top: -1`, `cam_side: +1`, `cam_flank: +1` |

---

## 🚀 Installation & Setup

```bash
# 1. Navigate to the app directory
cd apps/truck_loading_detection

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Option A: Web Dashboard (Recommended)

Start the interactive web dashboard on port `5000`:
```bash
python3 app.py
```
Open your browser at `http://localhost:5000` to upload/select two or three synchronized camera videos, configure each tripwire against the footage, and run recorded-video processing. The dashboard shows the camera feeds, red crossing lines, tracked carton boxes/local IDs in the processed output, and the central Loaded / Returned / Net counts. Confirmed event counts update during analysis; final counts are published only after processing completes. Recorded mode is independent of RTSP availability. Without spatial calibration, cross-camera matching relies on synchronized timing and direction, so verify all three image-space tripwires and validate results against manually established ground truth before operational use. Ambiguous or single-camera observations are excluded from the count and shown as diagnostic evidence.

In **Live Cameras**, enter each RTSP/HTTP URL or local USB device index (such as `0`), then connect to view the three feeds and their connection status. Camera URLs are not returned by the health/status APIs. Live automatic counting is not implemented; live preview does not imply live counting. Do not enable live counts until a synchronized live inference path is implemented and all three cameras have verified physical calibration.

### Collecting loading-gate calibration

Calibration is a separate setup page, not another counting dashboard. Open `/calibrate/<site_id>/<dock_id>` when collecting or validating measured camera geometry; routine counting and video review remain on `/`.

Open `/calibrate/<site_id>/<dock_id>` to calibrate `cam_top`, `cam_side`, and `cam_flank` one at a time. Capture a camera-specific live frame or upload a frame extracted from that camera's processed video. The frame must retain the exact processing resolution and native orientation.

At the warehouse, measure six named landmarks and four separate check points on the same physical loading-gate plane. Record every physical X/Y coordinate in meters from one shared origin and with the same axis direction for all cameras. Place the six fit points and four independent validation points on the frame, enter their measured coordinates, and use **Validate and save this camera**. A camera is saved only when its fit and independent reprojection errors pass the existing 2-pixel limits. The page shows the per-camera results and includes a placement guide; the guide is not a source of dimensions.

Do not enter estimated physical coordinates or treat one calibrated camera as a completed dock calibration. Physical calibration is optional for pixel-tripwire counting; it is required only when calibrated spatial association is used.

## Validation status

The model file hash and class contract have been verified. Unit/regression tests do not establish warehouse counting accuracy. Recorded-video evaluation must match annotated crossings and timing before the count is considered production-accurate; see the task's final audit results for the current known limitations.

---

## 🐳 Docker Deployment

```bash
docker build -t truck-loading-counter apps/truck_loading_detection
docker run -p 5000:5000 truck-loading-counter
```
