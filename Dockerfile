FROM python:3.12.6-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies for OCR, audio, and image processing.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies.
COPY backend/requirements.txt /app/backend/requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/backend/requirements.txt

# Copy the backend application.
COPY backend/app /app/backend/app
COPY backend/download_models.py /app/backend/download_models.py

# Copy the inference pipeline without committing model weights.
COPY ml-models/inference_pipelines /app/ml-models/inference_pipelines

WORKDIR /app/backend

EXPOSE 10000

# Download private models using the runtime environment variables,
# then start FastAPI on Render's assigned port.
CMD ["sh", "-c", "python download_models.py && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]