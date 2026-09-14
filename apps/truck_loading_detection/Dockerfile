FROM python:3.12-slim AS base

# Prevent Python from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for OpenCV + YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory with correct permissions
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# Non-root user for security
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

# Environment defaults
ENV PORT=5000 \
    HOST=0.0.0.0 \
    WORKERS=1 \
    TORCH_THREADS=4 \
    CONF_THRESHOLD=0.24 \
    IOU_THRESHOLD=0.45 \
    IMG_SIZE=416 \
    FRAME_STRIDE=2 \
    MAX_UPLOAD_MB=500 \
    JPEG_QUALITY=70 \
    VIDEO_SOURCE=""

EXPOSE 5000

# Health check against the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Production: use gunicorn. For single-worker real-time processing, 1 worker is correct
# (the inference thread is the bottleneck, not request concurrency).
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", \
     "--timeout", "120", "--keep-alive", "5", "--access-logfile", "-", \
     "--error-logfile", "-", "app:app", "--preload"]
