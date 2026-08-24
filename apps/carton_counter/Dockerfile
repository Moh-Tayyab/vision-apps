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

ENV MODEL_BACKEND=local
# Backends:
#   local            - ultralytics on-device (MODEL_PATH, CONF_THRESHOLD)
#   roboflow         - hosted model full-image inference (ROBOFLOW_MODEL_URL)
#   roboflow_workflow - SAHI sliced inference via saved workspace workflow
#                      (ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW; best accuracy,
#                      use CONF_THRESHOLD=0.10-0.11 - workflow returns all dets >= 0.01)
# Senior's YOLO26 recommendation; "start here" = yolo26m.
# Currently set to yolo26n (fastest); switch via MODEL_PATH=models/yolo26m.pt etc.
ENV MODEL_PATH=models/yolo26n.pt
ENV CONF_THRESHOLD=0.5
ENV ROBOFLOW_API_URL=https://serverless.roboflow.com
ENV ROBOFLOW_WORKSPACE=muhammad-tayyab-iqnwv
ENV ROBOFLOW_WORKFLOW=carton-counter-sahi
ENV ROBOFLOW_WORKFLOW_OUTPUT=stitched_predictions
ENV PORT=8000
ENV VIDEO_SOURCE=0
ENV STREAM_FPS=30

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
