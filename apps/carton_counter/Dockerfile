FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch first: avoids pulling the ~2GB CUDA wheel stack (laptop deployment)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p models \
    && curl -fsSL -o models/yolo26n.pt \
       https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt \
    && curl -fsSL -o models/yolo26m.pt \
       https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt

COPY . .

ENV MODEL_BACKEND=roboflow
# Backends:
#   roboflow          - RECOMMENDED: hosted RF-DETR v7 full-image inference
#                       (ROBOFLOW_MODEL_URL, CONF_THRESHOLD=0.36)
#   local             - ultralytics on-device (MODEL_PATH; offline fallback)
#   roboflow_workflow - SAHI sliced inference via saved workflow (legacy,
#                       for very dense scenes; CONF_THRESHOLD=0.11)
# Senior's YOLO26 recommendation for LOCAL backend; "start here" = yolo26m.
ENV MODEL_PATH=models/yolo26n.pt
ENV CONF_THRESHOLD=0.36
ENV ROBOFLOW_MODEL_URL=https://detect.roboflow.com/carton-counter-demo/7
ENV ROBOFLOW_API_URL=https://serverless.roboflow.com
ENV ROBOFLOW_WORKSPACE=muhammad-tayyab-iqnwv
ENV ROBOFLOW_WORKFLOW=carton-counter-sahi
ENV ROBOFLOW_WORKFLOW_OUTPUT=stitched_predictions
ENV PORT=8000
ENV VIDEO_SOURCE=0
ENV STREAM_FPS=30

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
