# ═══════════════════════════════════════════════════════════════════════
# JBDetection — Dockerfile
# ═══════════════════════════════════════════════════════════════════════
#
# Base:
#   CUDA 11.4.3 + cuDNN 8 + Ubuntu 20.04
#   Python 3.8
#
# PaddlePaddle:
#   Source: PaddlePaddle 2.4.2
#   Custom GPU build
#   CUDA: 11.4
#   GPU architecture: sm_37
#   Target GPU: NVIDIA Tesla K80
#
# IMPORTANT:
# The official PaddlePaddle 2.4.2 GPU wheel does not contain
# CUDA kernels for Tesla K80 (sm_37).
#
# Therefore this image uses the custom wheel built specifically
# for CUDA 11.4 + sm_37:
#
#   apps/backend/build_artifacts/paddle/
#       paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
#
# The wheel version appears as 0.0.0 because it was built directly
# from the PaddlePaddle 2.4.2 source tree.
#
# DO NOT replace this wheel with the official Paddle GPU wheel.
# ═══════════════════════════════════════════════════════════════════════


FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive


# ── 1. System packages + Python 3.8 ───────────────────────────────────

RUN rm -f /etc/apt/sources.list.d/*.list && \
    echo "deb http://archive.ubuntu.com/ubuntu focal main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu focal-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      ca-certificates \
      curl \
      wget \
      gnupg \
      lsb-release \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      python3-tk \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgomp1 \
      libpq5 \
      zlib1g \
      tzdata \
      poppler-utils \
      libpng16-16 \
      libjpeg8 \
      libtiff5 \
      libwebp6 \
      libopenjp2-7 \
      libleptonica-dev \
      libicu66 \
      libpango-1.0-0 \
      libcairo2 \
    && rm -rf /var/lib/apt/lists/*


# Make python3 / pip3 the default commands

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python3 -m pip install --upgrade pip "setuptools<70" wheel


# ── 2. Timezone ───────────────────────────────────────────────────────

ENV TZ=Asia/Tehran

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone


# ── 3. Create application directories + permissions ──────────────────

RUN mkdir -p \
        /home/devio/JB-outputs \
        /home/devio/JB-logs \
        /home/devio/.paddleocr \
        /home/devio/.cache && \
    chmod -R 777 /home/devio


# ── 4. Runtime environment ────────────────────────────────────────────

ENV PYTHONPATH=/app
ENV HOME=/home/devio
ENV XDG_CACHE_HOME=/home/devio/.cache

# Paddle/PyMuPDF/OpenCV runtime workaround.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libz.so.1

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

ENV TZ=Asia/Tehran

ENV DATABASE_URL=postgresql+psycopg2://vakav:vakav_strong_password_change_me@postgres:5432/vakav


# ── 4.1. CUDA/cuDNN compatibility symlinks ─────────────────────────────
#
# PaddlePaddle 2.4.2 expects the unversioned library names:
#
#   libcudnn.so
#   libcublas.so
#   libcublasLt.so
#
# The CUDA runtime image provides versioned libraries.
#
# Create the symlinks during IMAGE BUILD so that runtime containers
# do not require any manual library fixes.
#
# cuDNN:
#   /usr/lib/x86_64-linux-gnu/libcudnn.so.8
#
# cuBLAS:
#   /usr/local/cuda/lib64/libcublas.so.11
#   /usr/local/cuda/lib64/libcublasLt.so.11
#
# These are compatibility symlinks only.
# They do not change the CUDA or Paddle version.

RUN ln -sf \
    /usr/lib/x86_64-linux-gnu/libcudnn.so.8 \
    /usr/local/cuda/lib64/libcudnn.so && \
    ln -sf \
    /usr/local/cuda/lib64/libcublas.so.11 \
    /usr/local/cuda/lib64/libcublas.so && \
    ln -sf \
    /usr/local/cuda/lib64/libcublasLt.so.11 \
    /usr/local/cuda/lib64/libcublasLt.so


# ── 4.2. Verify CUDA runtime libraries exist in the image ─────────────
#
# Docker build normally has no GPU access.
# Therefore this only verifies that the required runtime libraries
# physically exist in the image.
#
# Actual GPU validation is performed after the container starts.

RUN test -e /usr/local/cuda/lib64/libcudnn.so && \
    test -e /usr/local/cuda/lib64/libcublas.so && \
    test -e /usr/local/cuda/lib64/libcublasLt.so && \
    echo "CUDA runtime library links: OK" && \
    ls -lah \
      /usr/local/cuda/lib64/libcudnn.so \
      /usr/local/cuda/lib64/libcublas.so \
      /usr/local/cuda/lib64/libcublasLt.so


# ── 5. Application working directory ──────────────────────────────────

WORKDIR /app


# ── 6. Install CUSTOM PaddlePaddle-GPU build ──────────────────────────
#
# IMPORTANT:
#
# Do NOT install:
#
#   paddlepaddle-gpu==2.4.2.post112
#
# from the official Paddle repository.
#
# That wheel does not contain Tesla K80 / sm_37 support.
#
# The wheel below was built from PaddlePaddle 2.4.2 source with:
#
#   CUDA 11.4
#   CUDA architecture: 37
#   Python 3.8
#   GPU enabled
#
# The same wheel was already validated inside jbdetection_test:
#
#   CUDA: True
#   Device: gpu:0
#   GPU: Tesla K80
#   Compute Capability: 3.7
#   CUDA Runtime: 11.4
#   cuDNN: 8.2
#
# It also passed a real PaddleOCR GPU inference test.

COPY apps/backend/build_artifacts/paddle/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl /tmp/


RUN pip3 install --no-cache-dir \
    /tmp/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl && \
    rm -f /tmp/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl


# ── 7. Install Python dependencies ────────────────────────────────────
#
# PaddleOCR and the remaining application dependencies are installed
# from the project's requirements file.
#
# PaddlePaddle itself is intentionally NOT installed from this file;
# the custom wheel above is the required GPU build.

COPY apps/backend/requirements-linux.txt /app/

RUN pip3 install --no-cache-dir --prefer-binary \
    -r requirements-linux.txt


# ── 8. Legacy TensorFlow/Keras compatibility layer ────────────────────
#
# These packages are temporarily retained because legacy compatibility
# modules/files still exist in the repository.
#
# Current target architecture uses PaddleOCR instead of:
#
#   Tesseract
#   Keras PDF classifier
#   TensorFlow
#
# However, removal is intentionally deferred until the migrated
# application has been fully validated through the real runtime.
#
# DO NOT use this section for PaddlePaddle.

RUN pip3 install --no-cache-dir --no-deps \
      tensorflow-gpu==2.10.0 \
      tensorflow-estimator==2.10.0 \
      keras==2.10.0 \
      Keras-Preprocessing==1.1.2 \
      h5py==3.11.0 \
      gast==0.4.0 \
      absl-py==2.3.1 \
      astunparse==1.6.3 \
      flatbuffers==25.2.10 \
      google-auth==2.40.3 \
      google-auth-oauthlib==0.4.6 \
      google-pasta==0.2.0 \
      grpcio==1.70.0 \
      libclang==18.1.1 \
      Markdown==3.4.4 \
      markdown-it-py==3.0.0 \
      mdurl==3.0.0 \
      ml_dtypes==0.2.0 \
      namex==0.1.0 \
      oauthlib==3.3.1 \
      pyasn1==0.6.1 \
      pyasn1_modules==0.4.2 \
      rsa==4.9.1 \
      termcolor==2.4.0 \
      wrapt==1.17.2 \
      tensorboard==2.10.1 \
      tensorboard-data-server==0.6.1 \
      tensorboard-plugin-wit==1.8.1 \
      tensorflow-io-gcs-filesystem==0.30.0 \
    || echo "WARN: Some TF packages failed to install (non-critical)"


# ── 9. Copy application source ────────────────────────────────────────

COPY apps /app/apps


# ── 10. Runtime volumes ───────────────────────────────────────────────

VOLUME ["/home/devio/JB-outputs", "/home/devio/JB-logs"]


# ── 11. Expose application port ───────────────────────────────────────

EXPOSE 5000


# ── 12. Start application ─────────────────────────────────────────────

CMD ["gunicorn", \
     "apps.backend.app:app", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "3600", \
     "--graceful-timeout", "3600", \
     "--keep-alive", "5", \
     "--worker-class", "gthread", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]