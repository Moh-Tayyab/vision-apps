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

ENV MODEL_BACKEND=local
# COCO yolo26n detects persons only (status=unknown).
# For helmet/no-helmet classes set MODEL_BACKEND=roboflow + ROBOFLOW_MODEL_URL,
# or point MODEL_PATH at local weights trained on person/helmet/head classes.
ENV MODEL_PATH=yolo26n.pt
ENV CONF_THRESHOLD=0.5
ENV PORT=8002

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
