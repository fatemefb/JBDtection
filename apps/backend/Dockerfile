FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    libgl1 \
    libglib2.0-0 \
    python3-dev \
    tesseract-ocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]

