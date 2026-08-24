FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_BACKEND=local
ENV MODEL_PATH=yolo11n.pt
ENV CONF_THRESHOLD=0.5
ENV PORT=8000
ENV VIDEO_SOURCE=0
ENV STREAM_FPS=30

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
