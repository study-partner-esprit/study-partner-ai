"""
Focus Detector Service
Integrates the ML focus detection model (MobileNetV2 transfer learning) with the API.
Classes: Focused (0), Drifting (1), Lost (2)
"""

import os
import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Model paths — prefer transfer-learned model, fall back to original
_SERVICE_DIR = Path(__file__).parent
_MODEL_TRANSFER = _SERVICE_DIR / "focus_model_transfer.h5"
_MODEL_ORIGINAL = _SERVICE_DIR / "focus_model.h5"
_MODEL_OUTPUTS = _SERVICE_DIR.parent.parent / "ML" / "focus" / "outputs" / "models"

LABEL_MAP = {0: "focused", 1: "drifting", 2: "lost"}
IMG_SIZE = (224, 224)


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

    def _resolve_model_path(self) -> Optional[Path]:
        """Find the best available model file."""
        candidates = [
            _MODEL_TRANSFER,
            _MODEL_ORIGINAL,
            _MODEL_OUTPUTS / "focus_model_transfer.h5",
            _MODEL_OUTPUTS / "focus_model.h5",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _load_model(self):
        """Load the TensorFlow/Keras focus detection model."""
        model_path = self._resolve_model_path()
        if model_path is None:
            logger.warning("No focus model file found — detector will return mock data")
            return

        try:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
            from tensorflow.keras.models import load_model  # noqa: lazy import

            self.model = load_model(str(model_path))
            self.model_loaded = True
            logger.info("Focus detector model loaded from %s", model_path)
        except Exception as e:
            logger.error("Failed to load focus model: %s", e)
            self.model_loaded = False

    def analyze_frame(self, frame_data: bytes) -> dict:
        """
        Analyze a video frame for focus state.
        Accepts raw JPEG/PNG bytes -> decodes -> resizes -> predicts.
        Returns mock data if model not loaded.
        """
        if not self.model_loaded or self.model is None:
            logger.warning("Focus model not loaded, returning mock data")
            return {
                "focus_score": 75.0,
                "focus_state": "focused",
                "confidence": 0.8,
                "error": "Model not loaded - using mock data",
            }

        try:
            import cv2  # noqa: lazy import

            # Decode bytes -> numpy -> resize -> normalise
            arr = np.frombuffer(frame_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes")

            img = cv2.resize(img, IMG_SIZE)
            img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

            preds = self.model.predict(img, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            confidence = float(np.max(preds))

            # Map to 0-100 score (Focused probability * 100)
            focus_score = float(preds[0] * 100)

            return {
                "focus_score": round(focus_score, 1),
                "focus_state": LABEL_MAP[pred_idx],
                "confidence": round(confidence, 3),
                "probabilities": {
                    LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(preds)
                },
            }
        except Exception as e:
            logger.error("Error analyzing frame: %s", e)
            return {
                "focus_score": 50.0,
                "focus_state": "unknown",
                "confidence": 0.0,
                "error": str(e),
            }

    def get_focus_state(self, score: float) -> str:
        if score >= 70:
            return "focused"
        elif score >= 40:
            return "drifting"
        else:
            return "lost"


_focus_detector = None


def get_focus_detector() -> FocusDetector:
    """Get or create the global focus detector instance."""
    global _focus_detector
    if _focus_detector is None:
        _focus_detector = FocusDetector()
    return _focus_detector
