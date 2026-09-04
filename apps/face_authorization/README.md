# App 3 — Production-Grade Face Authorization Microservice

High-throughput, real-time facial recognition and access control microservice built on **FastAPI**, **DeepFace (Facenet512 + RetinaFace)**, **Qdrant Vector Database**, and **SQLite WAL**.

Includes passive anti-spoofing liveness detection, temporal face tracking, Prometheus observability, and an administrative **Streamlit Dashboard**.

---

## Architecture Overview

```
[ CCTV / Mobile IP Camera / Video Stream ]
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Fast Face Detection (RetinaFace @ 14px min face size)        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Passive Anti-Spoofing Engine (<2ms multi-cue analysis)      │
│  - Chrominance Variance (YCrCb/HSV skin reflectance)        │
│  - Laplacian Texture / Sharpness Differential               │
│  - Fourier Spectrum Frequency Energy (Moiré pattern detector│
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 512-D Face Vector Embedding Extraction (Facenet512)         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Vector Database Similarity Match                            │
│  - Primary: Self-Hosted Qdrant Engine (HNSW Cosine Index)   │
│  - Fallback: SQLite WAL + Vectorized NumPy Cosine Search    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Multi-Target Temporal Face Tracker (IoU + Consensus Voting) │
│  - Exponential Moving Average Spatial Bounding Box Smoothing│
│  - Multi-frame Temporal Consensus Voting (No 1-frame jitter)│
└──────────────────────────┬──────────────────────────────────┘
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│ SQLite WAL Audit Trail       │ │ Prometheus Metrics Exporter  │
│ - Timestamped events         │ │ - /metrics for Grafana       │
│ - Violations & Spoof alarms  │ │ - Latency, FPS, Counts       │
└──────────────────────────────┘ └──────────────────────────────┘
```

---

## 🚀 Quickstart with Docker Compose (Recommended)

Start the entire self-hosted stack (Qdrant Vector DB + FastAPI Service + Streamlit UI) in 1 command:

```bash
cd apps/face_authorization
docker compose up --build -d
```

### Services & Ports:
- **FastAPI Core Backend**: [http://localhost:8003](http://localhost:8003)
- **FastAPI Interactive Docs**: [http://localhost:8003/docs](http://localhost:8003/docs)
- **Streamlit Admin UI**: [http://localhost:8501](http://localhost:8501)
- **Qdrant Vector Database**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- **Prometheus Metrics**: [http://localhost:8003/metrics](http://localhost:8003/metrics)

---

## 💻 Local Development (Zero Docker Requirement)

The service is fully resilient: if Qdrant is not running, it automatically uses **SQLite WAL + embedded NumPy vector math** with zero configuration required.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8003 --reload

# 3. In a separate terminal, start Streamlit Dashboard
streamlit run streamlit_app.py --server.port 8501
```

---

## 🛡️ Production Features

### 1. Distance & Small Face Sensitivity
- Resolution auto-upscaling (dynamic 1080p target) prevents distant faces from degrading.
- Minimum face size threshold lowered to **14px** (allows detection across warehouse distances).
- Tuned cosine distance threshold ($\le 0.48$) and RetinaFace detection confidence ($0.22$).

### 2. Passive Anti-Spoofing (Liveness Check)
- Evaluates skin chrominance variance, Fourier spectrum frequencies (detects screen refresh/print moiré), and Laplacian sharpness.
- Flags flat 2D presentation attacks (printed papers, iPad/phone screens) as `SPOOF_ATTACK`.

### 3. Multi-Target Temporal Tracking
- Prevents bounding-box flicker with exponential spatial smoothing ($\alpha = 0.70$).
- Eliminates 1-frame lighting false negatives using 6-frame majority consensus voting.

### 4. Enterprise Security & Auditability
- Optional API Key / Bearer token enforcement (`X-API-Key` or `Authorization: Bearer <token>`).
- Crash-safe SQLite WAL transaction logging for all authorization attempts, violations, and spoof alarms.
- Prometheus `/metrics` exporter compatible with Datadog, Prometheus, and Grafana.

---

## 📡 API Reference

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| `GET` | `/health` | Healthcheck and status | No |
| `GET` | `/metrics` | Prometheus metrics scrape target | No |
| `GET` | `/db/stats` | Database & vector index statistics | No |
| `GET` | `/audit/events` | Query recent authorization audit logs | Optional |
| `POST` | `/verify` | Single frame / image verification | Optional |
| `POST` | `/persons/enroll` | Enroll new authorized personnel | Yes (Admin) |
| `GET` | `/persons` | List all enrolled personnel | Optional |
| `DELETE` | `/persons/{name}` | Remove enrolled personnel | Yes (Admin) |
| `GET` | `/stream/detect` | Real-time annotated MJPEG camera stream | No |
| `POST` | `/ingest/frame` | Ingest frame from mobile/edge camera | No |
