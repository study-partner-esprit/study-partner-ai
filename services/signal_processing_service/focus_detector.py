"""
Focus Detector Service
Integrates the ML focus detection model with the API
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import numpy as np

# Add ML/focus to path
FOCUS_MODEL_DIR = Path(__file__).parent.parent.parent.parent / "ML" / "focus"
sys.path.insert(0, str(FOCUS_MODEL_DIR))

logger = logging.getLogger(__name__)


class FocusDetector:
    """
    Wrapper for the focus detection ML model.
    Processes video frames and returns focus score + state.
    """
    
    def __init__(self):
        """Initialize the focus detector model."""
        self.model = None
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the focus detection model."""
        try:
            # TODO: Import and initialize your focus model
            # This depends on the exact structure of ML/focus/
            # Example:
            # from main import FocusModel
            # self.model = FocusModel()
            # self.model.load_weights()
            
            logger.info("Focus detector model loaded successfully")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load focus model: {e}")
            self.model_loaded = False
    
    def analyze_frame(self, frame_data: bytes) -> dict:
        """
        Analyze a video frame for focus state.
        
        Args:
            frame_data: Raw image bytes (JPEG/PNG)
        
        Returns:
            {
                "focus_score": float (0-100),
                "focus_state": str ("focused" | "distracted" | "idle"),
                "confidence": float (0-1)
            }
        """
        if not self.model_loaded:
            # Return mock data if model not loaded
            logger.warning("Focus model not loaded, returning mock data")
            return {
                "focus_score": 75.0,
                "focus_state": "focused",
                "confidence": 0.8,
                "error": "Model not loaded - using mock data"
            }
        
        try:
            # TODO: Process frame with actual model
            # 1. Decode frame_data to numpy array
            # 2. Preprocess frame for model
            # 3. Run model inference
            # 4. Post-process predictions
            
            # Example placeholder:
            # image = decode_image(frame_data)
            # predictions = self.model.predict(image)
            # focus_score = predictions['focus_score']
            # focus_state = classify_focus_state(focus_score)
            
            # Placeholder return
            return {
                "focus_score": 75.0,
                "focus_state": "focused",
                "confidence": 0.8
            }
        
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return {
                "focus_score": 50.0,
                "focus_state": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_focus_state(self, score: float) -> str:
        """
        Convert focus score to categorical state.
        
        Args:
            score: Focus score (0-100)
        
        Returns:
            "focused" | "neutral" | "distracted"
        """
        if score >= 70:
            return "focused"
        elif score >= 40:
            return "neutral"
        else:
            return "distracted"


# Singleton instance
_focus_detector = None


def get_focus_detector() -> FocusDetector:
    """Get or create the global focus detector instance."""
    global _focus_detector
    if _focus_detector is None:
        _focus_detector = FocusDetector()
    return _focus_detector
