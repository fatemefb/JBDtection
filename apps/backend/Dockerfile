# ================================
# STAGE 1: Build OpenCV + Tesseract + Leptonica
# ================================
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

# ✅ بازگرداندن مخازن رسمی اوبونتو + اطمینان از اتصال
RUN rm -f /etc/apt/sources.list.d/*.list && \
    echo "deb http://archive.ubuntu.com/ubuntu focal main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu focal-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu focal-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update -o Acquire::Retries=5 -o Acquire::http::No-Cache=True -o Acquire::BrokenProxy=true && \
    apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates apt-transport-https curl wget gnupg lsb-release && \
    apt-get update && apt-get install -y \
      build-essential cmake git g++ \
      libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
      liblapack-dev libopenblas-dev pkg-config \
      python3.8 python3.8-dev python3-pip \
      autoconf automake libtool \
      libpng-dev libjpeg8-dev libtiff5-dev zlib1g-dev \
      libicu-dev libpango1.0-dev libcairo2-dev libgif-dev \
      libwebp-dev libopenjp2-7-dev \
      libleptonica-dev libgomp1 tzdata && \                                     
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip && pip install numpy

# ---------------------
# Leptonica
# ---------------------
WORKDIR /opt
RUN git clone https://github.com/DanBloomberg/leptonica.git && \
    cd leptonica && mkdir build && cd build && \
    cmake .. && make -j"$(nproc)" && make install && ldconfig

# ---------------------
# OpenCV
# ---------------------
WORKDIR /opt
RUN git clone https://github.com/opencv/opencv.git && \
    git clone https://github.com/opencv/opencv_contrib.git && \
    cd opencv && git checkout 4.8.1 && \
    cd ../opencv_contrib && git checkout 4.8.1

WORKDIR /opt/opencv/build
RUN PYTHON_BIN=$(which python) && \
    PYTHON_INCLUDE=$(python -c "from sysconfig import get_paths as gp; print(gp()['include'])") && \
    PYTHON_LIBDIR=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") && \
    PYTHON_SITEPKG=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") && \
    cmake \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=3.7 \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D BUILD_LIST=core,imgproc,highgui,videoio,python3,cudev,cudaimgproc,cudaarithm \
      -D BUILD_opencv_python3=ON \
      -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
      -D PYTHON3_EXECUTABLE=${PYTHON_BIN} \
      -D PYTHON3_INCLUDE_DIR=${PYTHON_INCLUDE} \
      -D PYTHON3_LIBRARY=${PYTHON_LIBDIR} \
      -D PYTHON3_PACKAGES_PATH=${PYTHON_SITEPKG} \
      .. && \
    make -j"$(nproc)" && make install && ldconfig

# ---------------------
# Tesseract
# ---------------------
WORKDIR /opt
RUN git clone https://github.com/tesseract-ocr/tesseract.git && \
    cd tesseract && git checkout 5.3.3 && mkdir build

WORKDIR /opt/tesseract/build
RUN cmake .. && make -j"$(nproc)" && make install && ldconfig

# دانلود traineddata
RUN mkdir -p /usr/local/share/tessdata && \
    curl -L -o /usr/local/share/tessdata/eng.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

ENV TESSDATA_PREFIX=/usr/local/share/tessdata
ENV TESSERACT_PATH=/usr/local/bin/tesseract

# ================================
# STAGE 2: Runtime
# ================================
FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/sources.list.d/*.list && \
    echo "deb http://archive.ubuntu.com/ubuntu focal main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu focal-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get clean && apt-get update && \
    apt-get install -y \
      python3.8 python3-pip tzdata \
      libpng16-16 libjpeg8 libtiff5 libwebp6 \
      libopenjp2-7 zlib1g libleptonica-dev \
      libgomp1 libicu66 libpango-1.0-0 libcairo2 \
      libgl1 libglib2.0-0 python3-tk \
      libpq5 curl \
      && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ✅ pip mirror
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.timeout 180

# ✅ copy libraries from builder
COPY --from=builder /usr/local /usr/local

# ✅ Create output dirs
RUN mkdir -p /home/devio/JB-outputs /home/devio/JB-logs && \
    chmod -R 777 /home/devio/JB-outputs /home/devio/JB-logs

# ✅ ✅ CRITICAL: Required runtime ENV
ENV TESSDATA_PREFIX=/usr/local/share/tessdata
ENV TESSERACT_PATH=/usr/local/bin/tesseract
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV TZ=Asia/Tehran
ENV DATABASE_URL=postgresql+psycopg2://vakav:vakav_password@postgres:5432/vakav

WORKDIR /app

COPY apps/backend/requirements-linux.txt /app/
RUN pip install --upgrade pip \
    && pip install --ignore-installed --prefer-binary --no-cache-dir -r requirements-linux.txt

COPY apps /app/apps

VOLUME ["/home/devio/JB-outputs", "/home/devio/JB-logs"]

CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:5000", "app:app"]

