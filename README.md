# Vision Apps (3 Independent FastAPI Services)

Three independent, production-ready vision services, each with its own FastAPI app and Dockerfile:

| App | Directory | Port | Purpose |
|---|---|---|---|
| **App 1** | `apps/carton_counter/` | **8001** | Carton counting on pallets (Live Video Stream + Multi-angle 3D Fusion) |
| **App 2** | `apps/helmet_detection/` | **8002** | Helmet vs no-helmet detection on live camera |
| **App 3** | `apps/face_authorization/` | **8003** | Authorized/unauthorized person via deepface embeddings |

---

## 🚀 Quick Start with Docker (Recommended)

### Run all 3 apps together:
```bash
docker compose up --build
```

### Run only App 1 (Carton Counter):
```bash
docker compose up --build carton-counter
```
Or build and run individually:
```bash
docker build -t carton-counter ./apps/carton_counter
docker run -p 8001:8000 --env-file .env carton-counter
```

---

## 📦 App 1: Carton Counter Live Demo

### 1. Laptop Monitor:
Open the live dashboard in your browser:
👉 **`http://localhost:8001/`**

### 2. Mobile Camera Streaming:
Open the mobile camera interface on your smartphone (connected to same network):
👉 **`http://<laptop-ip>:8001/mobile`** (or via Cloudflare / HTTPS tunnel)

- Tap **"Start Live Video Stream"** to stream continuous video from phone to server.
- The laptop screen will display the live video feed with **real-time green bounding boxes, `carton` labels, confidence scores, and total carton count**.

---

## 🧠 AI Detection & Architecture Highlights

- **Ensemble Validation:** Combines fine-tuned RF-DETR / Roboflow carton detection with a local COCO negative-class suppression model (YOLO) and geometric aspect-ratio filtering. This prevents non-carton household objects (e.g. pots, bags, cups, laptops, chairs) from being misclassified as cartons.
- **Multi-Angle Fusion:** `/count/multi-angle` endpoint takes Front, Side, and Top camera views of a pallet and computes the 3D volume count (`L × W × H` grid) with occlusion handling.
- **Lightweight Frame Buffer & MJPEG Streaming:** High-throughput `multipart/x-mixed-replace` video stream generator with frame deduplication and FPS throttling.

---

## ⚙️ Environment Configuration (`.env`)

Create a `.env` file in the root directory (or use `.env.example`):
```ini
MODEL_BACKEND=roboflow
ROBOFLOW_API_KEY=AydYTkTEwRNm3fM0n8yl
ROBOFLOW_MODEL_URL=https://detect.roboflow.com/carton-counter-demo/7
CONF_THRESHOLD=0.36
PORT=8001
```

---

## 🧪 Testing

Run smoke tests against the running server:
```bash
python tests/test_detector.py
```
