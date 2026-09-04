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
    && curl -fsSL -o yolo26n.pt \
       https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

COPY . .

ENV MODEL_BACKEND=roboflow
# Backends:
#   roboflow - RECOMMENDED: our hosted RF-DETR helmet model (head/helmet/person),
#              mAP@50=95.5, recall 96.4 (trained on 1088-image safety-helmet dataset)
#   local    - COCO yolo26n: persons only (status=unknown), offline fallback;
#              point MODEL_PATH at helmet-trained weights for full function
ENV ROBOFLOW_MODEL_URL=https://detect.roboflow.com/safety-helmet-dataset-uvh1t-aavk1/1
ENV MODEL_PATH=yolo26n.pt
ENV CONF_THRESHOLD=0.4
ENV PORT=8002

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
