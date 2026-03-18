"""
Fatigue Detector Service
Integrates rules-based fatigue detection (MediaPipe + FatigueRules) with the API.

Delegates to FatigueAdapter which uses:
  - MediaPipe FaceLandmarker for facial landmark extraction
  - FaceFeatures for EAR / MAR / blink / yawn computation
  - FatigueRules for rule-based fatigue scoring
"""

import logging
import cv2
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-load the adapter to avoid import crashes at module level
_adapter_class = None


def _get_adapter_class():
    """Lazily import FatigueAdapter to avoid startup crashes if deps are missing."""
    global _adapter_class
    if _adapter_class is None:
        try:
            from services.signal_processing_service.fatigue_adapter import (
                FatigueAdapter,
            )

            _adapter_class = FatigueAdapter
        except ImportError as e:
            logger.warning(f"FatigueAdapter not available: {e}")
            _adapter_class = False  # sentinel: tried and failed
    return _adapter_class if _adapter_class is not False else None


class FatigueDetector:
    """
    Wrapper around the rules-based FatigueAdapter.
    Decodes raw image bytes, forwards to the adapter, and returns
    a standardised dict consumed by the /analyze-frame endpoint.
    """

    def __init__(self):
        self.adapter = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        """Load the rules-based fatigue adapter."""
        try:
            AdapterCls = _get_adapter_class()
            if AdapterCls is not None:
                self.adapter = AdapterCls()
                self.model_loaded = self.adapter.is_model_loaded()
                if self.model_loaded:
                    logger.info("Fatigue detector (rules-based) loaded successfully")
                else:
                    logger.warning(
                        "FatigueAdapter instantiated but model not loaded (missing model file?)"
                    )
            else:
                logger.warning("FatigueAdapter unavailable – returning mock data")
        except Exception as e:
            logger.error(f"Failed to initialise fatigue adapter: {e}")
            self.model_loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_frame(self, frame_data: bytes) -> dict:
        """
        Analyse a video frame for fatigue indicators.

        Args:
            frame_data: Raw image bytes (JPEG / PNG).

        Returns:
            Dict with fatigue_score (0-100), fatigue_state, indicators, confidence.
        """
        if not self.model_loaded or self.adapter is None:
            logger.debug("Fatigue model not loaded – returning mock data")
            return self._mock_result()

        try:
            # Decode bytes → numpy BGR image
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Failed to decode frame data")
                return self._mock_result(error="Unable to decode frame")

            # Delegate to the rules-based adapter
            fatigue_state, fatigue_score, confidence = self.adapter.get_fatigue_signal(
                frame=frame
            )

            # Map score (0-1 float) → 0-100 int scale
            score_pct = round(fatigue_score * 100, 1)

            return {
                "fatigue_score": score_pct,
                "fatigue_state": fatigue_state,
                "indicators": {
                    "eye_closure": round(fatigue_score, 3),
                    "yawn_detected": fatigue_state in ("High", "Critical"),
                    "head_pose": {"pitch": 0, "yaw": 0, "roll": 0},
                },
                "confidence": round(confidence, 3),
            }

        except Exception as e:
            logger.error(f"Error analysing frame for fatigue: {e}", exc_info=True)
            return self._mock_result(error=str(e))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_result(error: Optional[str] = None) -> dict:
        result = {
            "fatigue_score": 15.0,
            "fatigue_state": "Alert",
            "indicators": {
                "eye_closure": 0.15,
                "yawn_detected": False,
                "head_pose": {"pitch": 0, "yaw": 0, "roll": 0},
            },
            "confidence": 0.0,
        }
        if error:
            result["error"] = error
        return result

    @staticmethod
    def get_fatigue_state(score: float) -> str:
        """Map a 0-100 score to a categorical label."""
        if score < 25:
            return "Alert"
        elif score < 50:
            return "Moderate"
        elif score < 75:
            return "High"
        return "Critical"


# ------------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------------
_fatigue_detector: Optional[FatigueDetector] = None


def get_fatigue_detector() -> FatigueDetector:
    """Get or create the global fatigue detector instance."""
    global _fatigue_detector
    if _fatigue_detector is None:
        _fatigue_detector = FatigueDetector()
    return _fatigue_detector
