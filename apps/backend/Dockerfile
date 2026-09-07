FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# PaddlePaddle 2.6.2 GPU wheels are published separately for CUDA 11.8.
ARG PADDLE_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Install system dependencies (بدون build tools — سریع‌تر)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev \
    libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender-dev \
    libpoppler-cpp-dev poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the backend requirements first (the compose build context is the repo root).
COPY apps/backend/requirements-linux.txt /app/requirements-linux.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        --index-url "${PADDLE_INDEX_URL}" \
        paddlepaddle-gpu==2.6.2 && \
    pip install --no-cache-dir -r /app/requirements-linux.txt

# Copy application code
COPY . /app/

# Set environment
ENV PYTHONPATH=/app
ENV TZ=Asia/Tehran
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libz.so.1

EXPOSE 5000

CMD ["gunicorn", "apps.backend.app:app", \
     "-b", "0.0.0.0:5000", \
     "--workers=2", \
     "--threads=4", \
     "--timeout=3600", \
     "--worker-class=sync", \
     "--log-level=info"]