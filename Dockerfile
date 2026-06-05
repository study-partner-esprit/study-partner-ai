# Use PyTorch base image (includes PyTorch, CUDA, cuDNN pre-installed)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    poppler-utils \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pymongo \
    pydantic \
    python-dotenv \
    google-generativeai \
    requests \
    PyMuPDF \
    pillow \
    faiss-cpu \
    scikit-learn \
    pytesseract \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scipy \
    pdf2image \
    PyPDF2 \
    reportlab \
    opencv-python \
    mediapipe \
    python-multipart \
    flask \
    beautifulsoup4 \
    apify-client \
    SpeechRecognition \
    pyttsx3 \
    pytest \
    pytest-asyncio \
    pytest-json-report \
    httpx \
    anyio \
    "transformers>=4.38.0,<4.45.0" \
    "sentence-transformers>=2.5.0,<2.8.0"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application with uvicorn
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
