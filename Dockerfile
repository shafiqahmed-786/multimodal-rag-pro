FROM python:3.11-slim

# System deps: Tesseract, poppler (for pdfplumber)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libgl1-mesa-glx \
    poppler-utils \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data/raw data/processed data/indexes .cache/embeddings

# Default: run the FastAPI backend
ENV PYTHONPATH=/app
EXPOSE 8000

CMD sh -c "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"
