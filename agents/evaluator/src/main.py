"""
Main entry point for the Socratic Evaluator application.
Launches the Gradio-based interactive UI.
No database required - fully offline and in-memory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging
from src.config.settings import LOG_LEVEL
from src.gradio_app import main as launch_gradio

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)


def main():
    """Main application entry point - launches Gradio UI."""
    try:
        logger.info("Starting Socratic Evaluator (Gradio)")
        logger.info("No database required - fully offline")
        launch_gradio()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise


if __name__ == "__main__":
    main()
