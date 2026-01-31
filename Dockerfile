# Use PyTorch base image (includes PyTorch, CUDA, cuDNN pre-installed)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install poetry and poetry export plugin
RUN pip install poetry poetry-plugin-export

# Copy dependency files first (for Docker layer caching)
COPY pyproject.toml poetry.lock* ./

# Export requirements, filter out packages already in base image, then install
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt \
    && grep -vE "^(torch|torchvision|torchaudio|nvidia|triton)" requirements.txt > requirements-filtered.txt \
    && pip install --no-cache-dir -r requirements-filtered.txt \
    && rm requirements.txt requirements-filtered.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')" || exit 1

# Run the application
CMD ["python", "main.py"]
