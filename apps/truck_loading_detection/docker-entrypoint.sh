#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Production Docker Entrypoint (Truck Loading Detection / Carton Counter)
# ---------------------------------------------------------------------------
# Pre-flight startup checks:
# 1. Validates canonical model checkpoint SHA-256 and task/class contracts.
# 2. Ensures persistent directories exist and initializes SQLite WAL EventStore.
# 3. Fails closed (exit 1) on any contract breach before serving any traffic.
# ---------------------------------------------------------------------------
set -eo pipefail

echo "=================================================="
echo "🚚 Truck Loading Detection - Container Startup"
echo "=================================================="

# 1. Directory creation
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve default db path: use /app/data if /app is writable (container environment),
# otherwise fallback to $APP_DIR/data (local/host environment).
if [ -n "${EVENT_STORE_DB_PATH:-}" ]; then
    DATA_PATH="$EVENT_STORE_DB_PATH"
elif [ -n "${CARTON_EVENTS_DB_PATH:-}" ]; then
    DATA_PATH="$CARTON_EVENTS_DB_PATH"
elif [ -w "/app" ] || [ -d "/app" ]; then
    DATA_PATH="/app/data/carton_events.db"
else
    DATA_PATH="$APP_DIR/data/carton_events.db"
fi

DATA_DIR="$(dirname "$DATA_PATH")"

mkdir -p "$DATA_DIR" "$APP_DIR/uploads" "$APP_DIR/output_videos" "$APP_DIR/config" 2>/dev/null || true
chmod -R 777 "$DATA_DIR" "$APP_DIR/uploads" "$APP_DIR/output_videos" 2>/dev/null || true

# 2. Canonical Model Checksum & Contract Validation
echo "[entrypoint] Pre-flight check: Validating canonical carton model..."
python3 - << 'EOF'
import sys
import os
import hashlib

MODEL_PATH = os.environ.get("CARTON_MODEL_PATH", "best.pt")
EXPECTED_SHA256 = "98c6164c83905feb9db6d9bee6f48ab239c5d68a00efbe0484dbf3448333ad33"

# Resolve model path
if not os.path.isfile(MODEL_PATH):
    for cand in ("/app/best.pt", os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")):
        if os.path.isfile(cand):
            MODEL_PATH = cand
            break

if not os.path.isfile(MODEL_PATH):
    print(f"[FATAL] Canonical model checkpoint not found: {MODEL_PATH}", file=sys.stderr)
    sys.exit(1)

# Check SHA-256 hash
hasher = hashlib.sha256()
with open(MODEL_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
        hasher.update(chunk)
calculated_sha = hasher.hexdigest()

if calculated_sha != EXPECTED_SHA256:
    print(f"[FATAL] Model SHA-256 hash mismatch!\n  Expected: {EXPECTED_SHA256}\n  Got:      {calculated_sha}", file=sys.stderr)
    sys.exit(1)

print(f"[entrypoint] Model SHA-256 verified successfully ({calculated_sha[:16]}...)")

# Check model architecture contract
try:
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    if getattr(model, "task", None) != "segment":
        print(f"[FATAL] Canonical model must be an instance segmentation model, got task='{model.task}'", file=sys.stderr)
        sys.exit(1)
    class_map = {int(k): str(v).lower() for k, v in model.names.items()}
    if class_map != {0: "carton"}:
        print(f"[FATAL] Canonical model must have exact class mapping {{0: 'carton'}}, got {class_map}", file=sys.stderr)
        sys.exit(1)
    print(f"[entrypoint] Model contract verified: task={model.task}, classes={class_map}")
except Exception as e:
    print(f"[FATAL] Failed to verify model contract: {e}", file=sys.stderr)
    sys.exit(1)
EOF

# 3. Durable SQLite WAL EventStore Initialization & Migration
echo "[entrypoint] Pre-flight check: Initializing SQLite WAL EventStore schema..."
python3 - << EOF
import os
import sys

app_dir = "$APP_DIR"
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from three_camera_system.event_store import EventStore

db_path = "$DATA_PATH"
try:
    store = EventStore(db_path=db_path)
    if store._corrupt:
        print(f"[FATAL] EventStore initialized in corrupt state at: {db_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[entrypoint] EventStore verified and schema initialized at: {db_path}")
except Exception as e:
    print(f"[FATAL] EventStore initialization error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo "[entrypoint] All pre-flight checks passed. Starting service command..."
echo "=================================================="

# Execute the primary container command
exec "$@"
