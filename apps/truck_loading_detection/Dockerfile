# ---------------------------------------------------------------------------
# Phase 4 — Production Dockerfile (Carton Counter / Truck Loading Detection)
# ---------------------------------------------------------------------------
# Builds a CPU-only production image. No GPU is assumed or required.
# Tested environment: Python 3.12, CPU-only torch, Debian-based slim base.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

# Prevent Python from buffering stdout/stderr (important for container logs).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for OpenCV (headless) + YOLO CPU inference.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Install pinned production dependencies first (cached layer).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (frozen models and configuration included).
COPY . .

# Startup validation entrypoint (runs frozen-model hash + config checks,
# then execs the main command).
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# ---------------------------------------------------------------------------
# Configuration version identification (baked at build time)
# ---------------------------------------------------------------------------
ARG BUILD_VERSION=dev
ARG MODEL_VERSION=unknown
ENV BUILD_VERSION=${BUILD_VERSION} \
    MODEL_VERSION=${MODEL_VERSION} \
    # Certified Phase 3.3 active configuration (preserved, not tuned using
    # locked-test data). Override at runtime via environment.
    PORT=5000 \
    HOST=0.0.0.0 \
    WORKERS=1 \
    TORCH_THREADS=4 \
    CONF_THRESHOLD=0.15 \
    CARTON_CONSENSUS_THRESHOLD=0.25 \
    IOU_THRESHOLD=0.45 \
    IMG_SIZE=640 \
    FRAME_STRIDE=2 \
    MAX_UPLOAD_MB=500 \
    JPEG_QUALITY=70 \
    VIDEO_SOURCE=output_videos/multi_angle_3angle_v2.mp4 \
    EVENT_STORE_DB_PATH=/app/data/carton_events.db \
    CARTON_EVENTS_DB_PATH=/app/data/carton_events.db \
    # Authoritative inventory is never written to by shadow mode; this flag is
    # a guard, not the sole protection (see shadow_mode implementation).
    AUTHORITATIVE_INVENTORY_LOCKED=true \
    # CORS: restrict to trusted origins in production (comma-separated).
    CORS_ORIGIN=localhost,127.0.0.1

# Persistent storage for SQLite event store and audit records.
VOLUME ["/app/data"]
RUN mkdir -p /app/data /app/uploads && chmod 777 /app/data /app/uploads

# Non-root user for security.
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

EXPOSE 5000

# Health check: application /health endpoint (includes frozen-model validity,
# detection/consensus thresholds, and model sha256).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; r=urllib.request.urlopen('http://localhost:5000/health'); import json; d=json.load(r); exit(0 if d.get('status')=='healthy' and d.get('models_valid') else 1)" || exit 1

# Stop gracefully (SIGTERM -> app shutdown handler -> state.cleanup()).
STOPSIGNAL SIGTERM

# Startup validation (frozen-model hash + config) runs in the entrypoint and
# fails closed before traffic is served.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
# Production: single gunicorn worker (inference thread is the bottleneck).
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", \
    "--timeout", "120", "--keep-alive", "5", "--graceful-timeout", "15", \
    "--access-logfile", "-", "--error-logfile", "-", "app:app", "--preload"]
