FROM python:3.11-slim

# System deps: tesseract for OCR, poppler for scanned-PDF-page OCR fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Data dirs — a persistent disk should be mounted at /app/data in production
# (see render.yaml) so indexes and uploaded files survive redeploys.
RUN mkdir -p data/raw data/processed

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
