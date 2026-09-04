FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    openssl \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# deepface model weights cache + enrolled embeddings persist here (mount a volume)
ENV DEEPFACE_HOME=/app/.deepface
ENV DATA_DIR=/app/data
ENV PORT=8003
ENV HTTPS_PORT=8445
RUN mkdir -p /app/.deepface /app/data /app/certs

EXPOSE 8003 8445

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
