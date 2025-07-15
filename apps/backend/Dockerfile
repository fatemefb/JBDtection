# Base image with CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install dependencies for OpenCV and Tesseract
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    libgl1 libglib2.0-0 \
    python3-dev python3-pip \
    ca-certificates libleptonica-dev \
    libicu-dev libpango1.0-dev libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install numpy (needed before building OpenCV)
RUN pip3 install numpy

# Clone and build OpenCV with CUDA support
WORKDIR /opt
RUN git clone https://github.com/opencv/opencv.git && \
    git clone https://github.com/opencv/opencv_contrib.git && \
    cd opencv && git checkout 4.8.1 && \
    cd ../opencv_contrib && git checkout 4.8.1

WORKDIR /opt/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D WITH_CUDA=ON \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_CUBLAS=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          .. && \
    make -j"$(nproc)" && make install && ldconfig

# ----------- Tesseract installation from GitHub ----------
# Clone tesseract
WORKDIR /opt
RUN git clone https://github.com/tesseract-ocr/tesseract.git && \
    cd tesseract && git checkout 5.3.3

# Build and install Tesseract
WORKDIR /opt/tesseract/build
RUN cmake .. && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig

# Download English traineddata
RUN mkdir -p /usr/local/share/tessdata && \
    curl -L -o /usr/local/share/tessdata/eng.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

# Set TESSDATA_PREFIX environment variable
ENV TESSDATA_PREFIX=/usr/local/share/

# --------------------- Python App Setup ------------------
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install --prefer-binary --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

