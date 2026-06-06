# Configuration settings for the Socratic Evaluator
# Fully offline, in-memory, no database required

import os
from pathlib import Path

# Google Gemini API Key (set via GEMINI_API_KEY env var)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or ""

# LM Studio (Qwen) settings - fallback for when Gemini is unavailable
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen2.5-7b-instruct")
LM_STUDIO_TIMEOUT = int(os.getenv("LM_STUDIO_TIMEOUT", "30"))
LM_STUDIO_TEMPERATURE = 0.3

# GPT4All settings
GPT4ALL_MODEL_NAME = "orca-mini-3b-gguf2-q4_0"  # Smaller, faster model
GPT4ALL_MODEL_PATH = os.getenv("GPT4ALL_MODEL_PATH")  # optional local path
GPT4ALL_TEMPERATURE = 0.2
GPT4ALL_DEVICE = "cpu"  # or "gpu"

# Logging
LOG_LEVEL = "INFO"

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
