# ===== Build Stage =====
CMD ["sh", "-c", "uvicorn app:app --host ${HOST} --port ${PORT}"]
# Run the application

    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
# Health check

EXPOSE ${PORT}
# Expose port

    PYTHONUNBUFFERED=1
    YOLO_MODEL_PATH=models/best.pt \
    DATA_DIR=data \
    MAX_VIDEO_SIZE_MB=500 \
    ALLOWED_ORIGINS=* \
    HOST=0.0.0.0 \
ENV PORT=8000 \
# Environment variables with defaults

RUN mkdir -p data/videos data/tracks data/annotations
# Create data directories

COPY models/ ./models/
COPY pipeline/ ./pipeline/
COPY app.py main.py ./
# Copy application code

ENV PATH=/root/.local/bin:$PATH
COPY --from=builder /root/.local /root/.local
# Copy Python packages from builder

    && apt-get clean
    && rm -rf /var/lib/apt/lists/* \
    ffmpeg \
    libgomp1 \
    libxrender-dev \
    libxext6 \
    libsm6 \
    libglib2.0-0 \
    libgl1-mesa-glx \
RUN apt-get update && apt-get install -y --no-install-recommends \
# Install runtime dependencies for OpenCV

WORKDIR /app

FROM python:3.11-slim
# ===== Production Stage =====


RUN pip install --no-cache-dir --user -r requirements.txt
# Install Python dependencies

COPY requirements.txt .
# Copy requirements first for better caching

    && rm -rf /var/lib/apt/lists/*
    build-essential \
RUN apt-get update && apt-get install -y --no-install-recommends \
# Install build dependencies

WORKDIR /app

FROM python:3.11-slim as builder

