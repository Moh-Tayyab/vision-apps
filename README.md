# 👁️ Enterprise Vision Apps Suite

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker)](https://www.docker.com/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v11%20%2F%20RF--DETR-FF6F00.svg)](https://ultralytics.com)
[![DeepFace](https://img.shields.io/badge/DeepFace-Facenet512-brightgreen.svg)](https://github.com/serengil/deepface)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)

An enterprise-grade, multi-service computer vision suite designed for industrial warehouse logistics, workplace safety enforcement, and biometric access control. The repository contains **3 decoupled, independent microservices**, each equipped with dedicated FastAPI backends, real-time MJPEG video streamers, mobile camera web interfaces, and modern analytical dashboards.

---

## 🏛️ System Architecture

The architecture is fully decoupled—each microservice operates in its own containerized environment with dedicated ports, independent AI models, and zero runtime cross-coupling:

```
                                    +------------------------------------------------------+
                                    |                INCOMING VIDEO SOURCES                |
                                    |  • Mobile Phones (Browser WebRTC / getUserMedia)     |
                                    |  • Wired USB Webcams (/dev/video*)                   |
                                    |  • RTSP / IP Cameras / Video Uploads                 |
                                    +------------------------------------------------------+
                                                              │
                     ┌────────────────────────────────────────┼────────────────────────────────────────┐
                     ▼                                        ▼                                        ▼
      +─────────────────────────────+          +─────────────────────────────+          +─────────────────────────────+
      |    APP 1: CARTON COUNTER    |          |   APP 2: HELMET DETECTION   |          |  APP 3: FACE AUTHORIZATION  |
      |   (Port 8001 / HTTPS 8443)  |          |   (Port 8002 / HTTPS 8444)  |          |   (Port 8003 / HTTPS 8445)  |
      +─────────────────────────────+          +─────────────────────────────+          +─────────────────────────────+
      | • Multi-View 2-Camera Fusion|          | • Real-time PPE Compliance  |          | • DeepFace / Facenet Engine |
      | • Layer-wise $N_1 \times N_2$ count|   | • Worker Helmet Verification|          | • Vector Face Embeddings    |
      | • RF-DETR + COCO Suppression|          | • Dynamic Violation Logging |          | • Streamlit Admin (Port 8501)|
      +─────────────────────────────+          +─────────────────────────────+          +─────────────────────────────+
                     │                                        │                                        │
                     ▼                                        ▼                                        ▼
      +─────────────────────────────+          +─────────────────────────────+          +─────────────────────────────+
      |  Dual-View Live Monitor UI  |          |  Safety Compliance Monitor  |          |  Live Gating + Streamlit UI |
      +─────────────────────────────+          +─────────────────────────────+          +─────────────────────────────+
```

---

## 📦 Microservices Breakdown

| Service | Directory | Port | Local SSL | Key Tech Stack | Primary Function |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **App 1: Carton Counter** | `apps/carton_counter/` | `8001` | `8443` | RF-DETR v7 / YOLOv11, OpenCV, Layer Fusion | 3D layer-by-layer pallet carton counting from dual camera views |
| **App 2: Helmet Detection** | `apps/helmet_detection/` | `8002` | `8444` | YOLOv8 / RF-DETR Safety Helmet Model | Real-time PPE safety compliance & violation tracking |
| **App 3: Face Authorization** | `apps/face_authorization/` | `8003` | `8445` | DeepFace, Facenet, RetinaFace, Streamlit | Biometric access control, user enrollment & live verification |
| **App 3 Admin UI** | `apps/face_authorization/` | `8501` | — | Streamlit Admin Portal | Employee photo gallery, user management & live detection monitor |

---

### 1. 📦 App 1: Carton Counter (Dual-Camera Layer-Wise Fusion)
* **The Problem:** Standard 2D bounding box counters fail on industrial pallets due to depth occlusions, stacked layers, and varying carton sizes.
* **The Solution:** Uses two synchronized camera perspectives:
  * **Front Camera (Cam 1):** Measures carton columns along the width and identifies physical layer heights ($N_{1,k}$).
  * **Side Camera at 90° (Cam 2):** Measures carton rows along the pallet depth for each corresponding layer ($N_{2,k}$).
* **Layer Calculation Formula:**
  $$\text{Layer } k \text{ Count} = N_{1,k} \times N_{2,k}$$
  $$\text{Total Pallet Count} = \sum_{k=1}^L (N_{1,k} \times N_{2,k})$$
* **AI Ensemble:** Combines fine-tuned RF-DETR carton detection ($mAP@50 = 97.9\%$) with negative-class COCO suppression and geometric aspect-ratio filtering to eliminate false positives from background clutter.

---

### 2. 🪖 App 2: Helmet Safety Detection (PPE Compliance)
* **Function:** Real-time safety compliance monitoring for construction sites, manufacturing plants, and warehouse loading docks.
* **Dual-Class Gating:** Simultaneously detects persons, heads, and safety hard hats. If a detected person lacks a verified helmet in their upper body region, the system triggers a **Safety Violation**.
* **Visual & Event Logging:** Real-time visual indicator (🟩 Green for compliant workers, 🟥 Red for violations), with continuous audit logging on `/violations`.

---

### 3. 🔐 App 3: Face Authorization & Access Control
* **Biometric Engine:** Generates 512-dimensional facial feature vectors using Facenet / RetinaFace backends with cosine similarity distance thresholding.
* **FastAPI Microservice (Port `8003`):** High-throughput frame ingestion, real-time identity verification, and live gate access granting/denial.
* **Streamlit Admin Portal (Port `8501`):** Complete management portal featuring:
  * 📊 **Analytics Dashboard:** Live stats on enrolled personnel and security access events.
  * ➕ **Face Enrollment:** One-click registration of employee name and frontal reference photo.
  * 👥 **User Management:** Photo ID card gallery with instant profile revocation/deletion.
  * 📹 **Live Gate Monitor:** Multi-source verification feed (webcam, IP camera, image upload).

---

## 🚀 Quick Start Guide

### Prerequisites
* [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/) installed
* Python 3.12+ (if running without Docker)
* A valid Roboflow API key in `.env` (optional, local YOLO weights provided as fallback)

---

### Option A: Launch All Services in 1 Command *(Recommended)*

Run the universal launcher script to build all containers and create secure Cloudflare HTTPS tunnels:

```bash
./start.sh
```

The script will launch all containers and display live access links:
* 📦 **Carton Counter Dashboard:** `http://localhost:8001/` (or Cloudflare HTTPS URL)
* 🪖 **Helmet Safety Dashboard:** `http://localhost:8002/` (or Cloudflare HTTPS URL)
* 🔐 **Face Authorization Live Gate:** `http://localhost:8003/` (or Cloudflare HTTPS URL)
* 📊 **Face Admin Streamlit Portal:** `http://localhost:8501/`

---

### Option B: Docker Compose (Standard)

```bash
# Build and run all 4 containers in the background
docker compose up --build -d

# View container logs
docker compose logs -f
```

To stop all containers:
```bash
docker compose down
```

---

### Option C: Local Virtual Environment (Without Docker)

```bash
# 1. Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies for all apps
pip install -r apps/carton_counter/requirements.txt
pip install -r apps/helmet_detection/requirements.txt
pip install -r apps/face_authorization/requirements.txt
pip install -r apps/face_authorization/requirements-streamlit.txt

# 3. Launch App 1 (Carton Counter)
PYTHONPATH=. python apps/carton_counter/main.py

# 4. Launch App 2 (Helmet Detection - in a new terminal)
PYTHONPATH=. python apps/helmet_detection/main.py

# 5. Launch App 3 (Face Authorization - in a new terminal)
PYTHONPATH=. python apps/face_authorization/main.py

# 6. Launch Streamlit Admin Portal (in a new terminal)
streamlit run apps/face_authorization/streamlit_app.py --server.port 8501
```

---

## 📱 Mobile Camera Streaming

All three applications feature dedicated browser-based mobile camera streaming interfaces (`/mobile`) that run directly on iOS Safari and Android Chrome without requiring any mobile app installations.

> 📖 **For complete step-by-step instructions on connecting mobile phones, setting up dual cameras, configuring local Wi-Fi SSL, and adjusting camera permissions, see the [Multi-Camera Connection & Usage Guide (GUIDE.md)](./GUIDE.md).**

---

## 📡 REST API Reference

All microservices provide interactive OpenAPI / Swagger documentation at `/docs`.

### App 1: Carton Counter (`http://localhost:8001`)

| Method | Endpoint | Description | Payload / Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health status and stream buffer state | — |
| `GET` | `/model/info` | Loaded model details and active backend | — |
| `POST` | `/detect` | Single image carton detection | Multipart `file: Image`, `confidence: float` |
| `POST` | `/detect/visualize` | Single image detection returning annotated image | Multipart `file: Image`, `confidence: float` |
| `POST` | `/count/dual` | Dual-camera layer-wise carton counting | Multipart `front: Image`, `side: Image`, `confidence: float` |
| `POST` | `/ingest` | Push real-time video frame into buffer | Multipart `file: JPEG Frame`, `cam: cam1 \| cam2` |
| `GET` | `/stream/detect` | Annotated MJPEG live video feed | — |
| `GET` | `/usb/cameras` | List connected V4L2 USB camera devices | — |

---

### App 2: Helmet Safety Detection (`http://localhost:8002`)

| Method | Endpoint | Description | Payload / Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health status and camera buffer metrics | — |
| `GET` | `/model/info` | Helmet detection model architecture | — |
| `POST` | `/detect` | Detect persons, helmets, and safety violations | Multipart `file: Image`, `confidence: float` |
| `POST` | `/detect/visualize` | Returns annotated JPEG with colored bounding boxes | Multipart `file: Image` |
| `POST` | `/ingest` | Push mobile camera frame into detection stream | Multipart `file: JPEG Frame` |
| `GET` | `/violations` | List recent safety violation audit records | `limit: int` |
| `GET` | `/stream/detect` | Real-time annotated MJPEG compliance stream | — |

---

### App 3: Face Authorization (`http://localhost:8003`)

| Method | Endpoint | Description | Payload / Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | System health & total enrolled users count | — |
| `POST` | `/enroll` | Register a new user with face embedding & photo | Multipart `name: str`, `file: Image` |
| `GET` | `/persons` | List all enrolled personnel | — |
| `GET` | `/persons/{name}/photo` | Retrieve stored enrollment photo | Path `name: str` |
| `DELETE`| `/persons/{name}` | Remove user, embeddings, and stored photo | Path `name: str` |
| `POST` | `/verify` | Match face against enrolled database | Multipart `file: Image`, `threshold: float` |
| `POST` | `/ingest` | Push live camera frame for real-time gating | Multipart `file: JPEG Frame` |
| `GET` | `/events` | Access log history (Authorized vs Intruder) | `limit: int` |
| `GET` | `/stream/detect` | Live annotated access control video stream | — |

---

## ⚙️ Environment Configuration (`.env`)

Create a `.env` file in the root directory (see `.env.example`):

```ini
# Global Roboflow API Key
ROBOFLOW_API_KEY=your_roboflow_api_key_here
MODEL_BACKEND=roboflow

# App 1: Carton Counter Configuration
CARTON_PORT=8001
CARTON_ROBOFLOW_MODEL_URL=https://detect.roboflow.com/carton-counter-demo/7
CARTON_CONF_THRESHOLD=0.36

# App 2: Helmet Detection Configuration
HELMET_PORT=8002
HELMET_ROBOFLOW_MODEL_URL=https://detect.roboflow.com/safety-helmet-dataset-uvh1t-aavk1/1
HELMET_CONF_THRESHOLD=0.30

# App 3: Face Authorization Configuration
FACE_PORT=8003
```

---

## 📂 Project Structure

```
carton-counter/
├── apps/
│   ├── carton_counter/               # App 1: Dual-camera carton counter
│   │   ├── main.py                   # FastAPI server & endpoints
│   │   ├── detector.py               # YOLO & RF-DETR carton detection engine
│   │   ├── counter.py                # Single & multi-view counting algorithms
│   │   ├── layer_counter.py          # Vertical layer clustering & de-duplication
│   │   ├── dual_fusion_engine.py     # 2-camera (Front + Side) layer fusion
│   │   ├── streamer.py               # FrameBuffer & MJPEG stream generator
│   │   ├── Dockerfile                # Standalone container specification
│   │   └── requirements.txt          # Python dependencies
│   │
│   ├── helmet_detection/             # App 2: PPE safety compliance
│   │   ├── main.py                   # FastAPI server, UI & violation log
│   │   ├── detector.py               # Person & helmet dual-detector engine
│   │   ├── streamer.py               # Video frame streaming pipeline
│   │   ├── Dockerfile                # Standalone container specification
│   │   └── requirements.txt          # Python dependencies
│   │
│   └── face_authorization/           # App 3: Biometric access control
│       ├── main.py                   # FastAPI backend & verification engine
│       ├── face_engine.py            # DeepFace / Facenet embedding store
│       ├── camera_manager.py         # Multi-source camera manager (USB/RTSP/IP)
│       ├── streamlit_app.py          # Admin Management Dashboard (Port 8501)
│       ├── Dockerfile                # FastAPI container specification
│       ├── Dockerfile.streamlit      # Streamlit container specification
│       └── requirements.txt          # Python dependencies
│
├── docker-compose.yml                # Multi-container orchestration specification
├── GUIDE.md                          # Comprehensive camera connection & setup manual
├── README.md                         # Project documentation
├── start.sh                          # Universal 1-command launcher (Docker + Cloudflare)
├── start_tunnel.sh                   # App 1 tunnel launcher
├── start_wired.sh                    # Wired USB camera launcher
└── tests/                            # Automated test suite
```

---

## 🧪 Testing & Validation

Execute the test suite to verify detector accuracy, API endpoints, and layer fusion math:

```bash
# Run detector unit tests
python tests/test_detector.py

# Run layer counter verification tests
pytest tests/test_layer_counter.py -v
```

---

## 🔒 Security & Production Guidelines

1. **Camera Permissions:** Mobile streaming strictly requires HTTPS. In production, always deploy behind an SSL reverse proxy (Nginx, Traefik, or Cloudflare).
2. **Data Privacy:** Face embeddings are stored locally as non-reversible mathematical vectors in `apps/face_authorization/data/embeddings.json`. Reference photos are stored in volume-isolated storage.
3. **Hardware Acceleration:** For high-density industrial deployments, enable GPU acceleration in `docker-compose.yml` (`nvidia-container-runtime`).
