# Configuration settings for the Socratic Evaluator
# Fully offline, in-memory, no database required

import os
from pathlib import Path

# Google Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAdvG4V9kEktHPlxfhlHwxG_I7xlE3Fs08")

# GPT4All settings
GPT4ALL_MODEL_NAME = "orca-mini-3b-gguf2-q4_0"  # Smaller, faster model
GPT4ALL_MODEL_PATH = os.getenv("GPT4ALL_MODEL_PATH")  # optional local path
GPT4ALL_TEMPERATURE = 0.2
GPT4ALL_DEVICE = "cpu"  # or "gpu"

# Logging
LOG_LEVEL = "INFO"

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
