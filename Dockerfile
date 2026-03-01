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

# Install Python packages that aren't in the base image
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
    sentence-transformers \
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
    # Search agent deps \
    flask \
    beautifulsoup4 \
    apify-client \
    SpeechRecognition \
    pyttsx3 \
    # Testing toolchain \
    pytest \
    pytest-asyncio \
    pytest-json-report \
    httpx \
    anyio

# Install poetry for dependency management
RUN pip install --no-cache-dir poetry poetry-plugin-export

# Copy dependency files first (for Docker layer caching)
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not create virtual env (we're already in a container)
RUN poetry config virtualenvs.create false

# Install any additional dependencies from poetry (if needed)
RUN poetry install --no-root --no-dev --no-interaction --no-ansi \
    && pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application with uvicorn
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
