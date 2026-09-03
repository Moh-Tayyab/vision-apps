# 🚚 Real-Time Truck Loading Carton & Worker Tripwire Counter

A high-performance computer vision system for tracking and counting cartons and workers loading goods into delivery trucks. Built using **Python**, **YOLO (Ultralytics)**, **ByteTrack**, **Flask**, and **OpenCV**.

---

## 🎯 Features

- **Directional Tripwire Counting**:
  - **Left-to-Right (+1 / Loaded)**: When worker/carton moves towards the truck.
  - **Right-to-Left (-1 / Returned)**: When worker/carton moves back from the truck.
- **Robust Multi-Object Tracking**: Utilizes **ByteTrack** for persistent ID association across frames and occlusions.
- **Hysteresis & Anti-Double Counting**: Built-in spatial hysteresis zone and frame cooldown preventing noisy double counting at the line boundary.
- **Dual Interfaces**:
  - **Interactive Web Dashboard (Flask)**: Live video streaming with responsive controls, real-time metrics, line adjustments, and upload capabilities.
  - **CLI / OpenCV GUI Runner**: High-performance desktop and headless video processing engine.
- **Interactive Tripwire Placement**:
  - Click and drag the virtual vertical line directly on the screen or dashboard.
  - Keyboard shortcuts for fine adjustments.
- **Versatile Input/Output**: Works with MP4/AVI videos, live USB webcams, RTSP IP cameras, and headless edge servers.

---

## 📁 Project Structure

```text
apps/truck_loading_detection/
├── app.py                      # Interactive Flask web application & live streaming backend
├── counter_engine.py           # Core directional tripwire logic and state machine
├── visualizer.py               # HUD overlay, glow line, trails, and detection renderer
├── main.py                     # CLI application entrypoint
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerization specification
├── templates/
│   └── index.html              # Modern Web Dashboard UI
├── sample_truck_loading.mp4    # Sample warehouse loading video
└── README.md                   # Documentation
```

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
Open your browser at `http://localhost:5000` to adjust confidence, IoU, line position, upload videos, and monitor live counts.

### Option B: CLI / OpenCV Window

#### 1. Run on Sample Video
```bash
python3 main.py --source sample_truck_loading.mp4 --save
```

#### 2. Run on Webcam or Live Camera
```bash
python3 main.py --source 0
```

#### 3. Run with a Custom Line Position (e.g. X = 510)
```bash
python3 main.py --source sample_truck_loading.mp4 --line-x 510 --save
```

#### 4. Run with a Custom Trained Carton YOLO Model
```bash
python3 main.py --source sample_truck_loading.mp4 --model path/to/custom_carton.pt --classes carton box
```

#### 5. Headless Mode (for Cloud/Server Background Processing)
```bash
python3 main.py --source sample_truck_loading.mp4 --no-show --output output_counted.mp4
```

---

## 🎮 Interactive Controls (CLI Mode)

| Key / Action | Function |
| :--- | :--- |
| **Mouse Drag on Line** | Reposition vertical tripwire live on the video feed |
| **`Left` / `Right` Arrow** or **`-` / `+`** | Nudge line left or right by 10 pixels |
| **`SPACE`** | Pause / Resume playback |
| **`R`** | Reset counters and event history |
| **`S`** | Save high-resolution annotated screenshot (`snapshot_frame_XXXX.png`) |
| **`Q`** / **`ESC`** | Quit application |

---

## ⚙️ CLI Options & Parameters

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | string / int | `sample_truck_loading.mp4` | Video file, RTSP stream URL, or webcam index (`0`) |
| `--model` | string | `yolov8n.pt` | YOLO weights (`yolov8n.pt`, `yolov8s.pt`, custom `.pt`) |
| `--line-x` | int | `width // 2` | Vertical line X coordinate |
| `--conf` | float | `0.30` | Detection confidence threshold |
| `--iou` | float | `0.50` | Tracker IoU / NMS threshold |
| `--classes` | list | `person`, `carton`, `box` | Filter specific class names or IDs |
| `--tracker` | string | `bytetrack.yaml` | Tracker config (`bytetrack.yaml` or `botsort.yaml`) |
| `--hysteresis` | int | `15` | Pixel margin buffer around tripwire |
| `--cooldown` | int | `15` | Minimum frame cooldown between crossings |
| `--save` | flag | `False` | Save annotated video output |
| `--output` | string | `output_truck_loading.mp4`| Output video filepath |
| `--no-show` | flag | `False` | Run in headless mode without GUI window |

---

## 🐳 Docker Deployment

```bash
docker build -t truck-loading-counter apps/truck_loading_detection
docker run -p 5000:5000 truck-loading-counter
```
