"""
Integration tests for the signal processing pipeline.

Tests the end-to-end flow:
  fatigue_detector.py  →  fatigue_adapter.py  →  FatigueRules / FaceFeatures
  focus_detector.py    →  focus_adapter.py    →  CNN model

These tests verify that the detectors initialise and can process
a synthetic frame without crashing, even when the ML model files
are not present (graceful fallback to mock data).
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Fatigue Detector ──────────────────────────────────────────────


class TestFatigueDetector:
    """Integration test for fatigue_detector → fatigue_adapter chain."""

    def test_singleton_creation(self):
        from services.signal_processing_service.fatigue_detector import (
            get_fatigue_detector,
        )

        det1 = get_fatigue_detector()
        det2 = get_fatigue_detector()
        assert det1 is det2, "get_fatigue_detector should return the same singleton"

    def test_analyze_frame_returns_required_keys(self):
        from services.signal_processing_service.fatigue_detector import (
            get_fatigue_detector,
        )

        detector = get_fatigue_detector()

        # Create a synthetic 640×480 BGR image (black)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Encode to JPEG bytes
        import cv2

        _, buf = cv2.imencode(".jpg", dummy_frame)
        frame_bytes = buf.tobytes()

        result = detector.analyze_frame(frame_bytes)

        assert isinstance(result, dict)
        assert "fatigue_score" in result
        assert "fatigue_state" in result
        assert "confidence" in result
        assert "indicators" in result
        assert isinstance(result["fatigue_score"], (int, float))
        assert 0 <= result["fatigue_score"] <= 100

    def test_analyze_invalid_bytes(self):
        from services.signal_processing_service.fatigue_detector import (
            get_fatigue_detector,
        )

        detector = get_fatigue_detector()

        result = detector.analyze_frame(b"not-an-image")
        assert isinstance(result, dict)
        # Should gracefully return mock/error data
        assert "fatigue_score" in result

    def test_fatigue_state_mapping(self):
        from services.signal_processing_service.fatigue_detector import FatigueDetector

        assert FatigueDetector.get_fatigue_state(10) == "Alert"
        assert FatigueDetector.get_fatigue_state(30) == "Moderate"
        assert FatigueDetector.get_fatigue_state(60) == "High"
        assert FatigueDetector.get_fatigue_state(80) == "Critical"


# ── Focus Detector ────────────────────────────────────────────────


class TestFocusDetector:
    """Integration test for focus_detector."""

    def test_singleton_creation(self):
        from services.signal_processing_service.focus_detector import get_focus_detector

        det1 = get_focus_detector()
        det2 = get_focus_detector()
        assert det1 is det2

    def test_analyze_frame_returns_required_keys(self):
        from services.signal_processing_service.focus_detector import get_focus_detector

        detector = get_focus_detector()

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        import cv2

        _, buf = cv2.imencode(".jpg", dummy_frame)
        frame_bytes = buf.tobytes()

        result = detector.analyze_frame(frame_bytes)

        assert isinstance(result, dict)
        assert "focus_score" in result
        assert "focus_state" in result
        assert "confidence" in result


# ── Signal Processing Service ────────────────────────────────────


class TestSignalProcessingService:
    """Integration test for the orchestrating SignalProcessingService."""

    def test_service_init_does_not_crash(self):
        from services.signal_processing_service.service import SignalProcessingService

        svc = SignalProcessingService()
        # Should always succeed — adapters may or may not load
        assert svc is not None

    def test_service_produces_snapshot(self):
        from services.signal_processing_service.service import SignalProcessingService

        svc = SignalProcessingService()
        snapshot = svc.get_current_signal_snapshot(user_id="test-user-integration")
        assert snapshot is not None
        assert hasattr(snapshot, "focus_state")
        assert hasattr(snapshot, "fatigue_state")
        assert hasattr(snapshot, "user_id")
        assert snapshot.user_id == "test-user-integration"
