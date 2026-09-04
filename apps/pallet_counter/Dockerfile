FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hosted inference only -- no local model weights, so the image stays small.
ENV ROBOFLOW_MODEL_URL=https://detect.roboflow.com/carton-counter-demo/7
ENV CONF_THRESHOLD=0.36
ENV PORT=8004

EXPOSE 8004

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
