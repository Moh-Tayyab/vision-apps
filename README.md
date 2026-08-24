# Carton Counter

Count cartons on pallet using YOLO detection with multi-angle fusion.

## Project Structure

```
carton-counter/
├── apps/
│   └── carton_counter/
│       ├── main.py              # FastAPI application
│       ├── detector.py          # YOLO detection engine
│       ├── counter.py           # Counting logic
│       ├── models/              # YOLO weights
│       │   └── yolo11n.pt       # COCO pre-trained
│       ├── requirements.txt
│       └── Dockerfile
├── tests/
│   └── test_detector.py
├── .env                         # API keys
├── .gitignore
└── opencode.jsonc               # Opencode config with Roboflow MCP
```

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
cd apps/carton_counter
pip install -r requirements.txt
```

### 3. Run Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Test API

```bash
# Health check
curl http://localhost:8001/health

# Detect cartons
curl -X POST http://localhost:8001/detect \
  -F "file=@image.jpg" \
  -F "confidence=0.5"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/detect` | Single image detection |
| POST | `/count` | 3-angle merged count |
| POST | `/count/video` | Video processing |
| POST | `/detect/visualize` | Detection with visualization |

## Docker

```bash
docker build -t carton-counter .
docker run -p 8001:8000 carton-counter
```

## Opencode

Open this directory with opencode to use Roboflow MCP server:

```bash
cd /home/muhammadtayyab/projects/carton-counter
opencode
```
