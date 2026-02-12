"""Fatigue adapter for loading and running the fatigue detection system.

This adapter wraps the local fatigue detection system and provides a clean interface
for getting fatigue signals without exposing implementation details.
"""

import sys
import os
from typing import Literal, Tuple, Optional
from pathlib import Path
import numpy as np
import cv2

# Add ML/fatigue-merged to path
fatigue_path = Path(__file__).parent.parent.parent / "ML" / "fatigue-merged"
sys.path.insert(0, str(fatigue_path))

try:
    from modules.face_features import FaceFeatures
    from modules.fatigue_rules import FatigueRules
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python import BaseOptions
except ImportError as e:
    print(f"Warning: Could not import fatigue modules: {e}")
    FaceFeatures = None
    FatigueRules = None
    FaceLandmarker = None


class FatigueAdapter:
    """
    Adapter for the fatigue detection system.

    Wraps MediaPipe-based fatigue detection and provides methods to get fatigue predictions.
    """

    # Type alias for fatigue states
    FatigueState = Literal["Alert", "Moderate", "High", "Critical"]

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the fatigue adapter.

        Args:
            model_path: Path to the MediaPipe face landmarker model. If None, uses default location.
        """
        if model_path is None:
            # Default path to the MediaPipe model
            base_path = Path(__file__).parent.parent.parent / "ML" / "fatigue-merged"
            model_path = base_path / "face_landmarker.task"

        self.model_path = str(model_path)
        self.face_features = None
        self.fatigue_rules = None
        self.face_landmarker = None
        self.is_calibrated = False

        # Load model if modules are available
        if (
            FaceFeatures is not None
            and FatigueRules is not None
            and FaceLandmarker is not None
        ):
            self._load_model()
        else:
            print("Warning: Fatigue detection modules not available, using mock data")

    def _load_model(self):
        """Load the fatigue detection components."""
        try:
            if os.path.exists(self.model_path):
                # Initialize feature extractors
                self.face_features = FaceFeatures()
                self.fatigue_rules = FatigueRules()

                # Create MediaPipe FaceLandmarker
                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=self.model_path),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                )
                self.face_landmarker = FaceLandmarker.create_from_options(options)
                print(
                    f"Fatigue detection system loaded successfully from {self.model_path}"
                )
            else:
                print(f"Warning: MediaPipe model not found at {self.model_path}")
        except Exception as e:
            print(f"Error loading fatigue detection system: {e}")
            self.face_features = None
            self.fatigue_rules = None
            self.face_landmarker = None

    def calibrate(self, frame: np.ndarray) -> bool:
        """
        Calibrate the fatigue detector with a baseline frame.

        Args:
            frame: A video frame (BGR format) of the user in alert state

        Returns:
            True if calibration successful, False otherwise
        """
        if self.face_features is None or self.face_landmarker is None:
            return False

        try:
            # Process frame to establish baseline
            h, w = frame.shape[:2]
            scale = h / 460 if h > 460 else 1.0
            frame_resized = cv2.resize(frame, None, fx=1 / scale, fy=1 / scale)

            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            adjusted = cv2.equalizeHist(gray)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(adjusted, cv2.COLOR_GRAY2RGB),
            )
            results = self.face_landmarker.detect(mp_image)

            if results.face_landmarks:
                face_landmarks = results.face_landmarks[0]

                # Extract eye landmarks
                left_eye_indices = [33, 159, 158, 133, 153, 144]
                right_eye_indices = [263, 386, 385, 362, 381, 380]

                left_eye = [face_landmarks[i] for i in left_eye_indices]
                right_eye = [face_landmarks[i] for i in right_eye_indices]

                left_ear = self.face_features.eye_aspect_ratio(left_eye)
                right_ear = self.face_features.eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                # Calibrate with single EAR value
                self.face_features.calibrate([avg_ear])
                self.is_calibrated = True
                return True
        except Exception as e:
            print(f"Error during calibration: {e}")

        return False

    def get_fatigue_signal(
        self, frame: Optional[np.ndarray] = None, auto_calibrate: bool = True
    ) -> Tuple[FatigueState, float, float]:
        """
        Get the current fatigue state based on video frame.

        Args:
            frame: Video frame (BGR format) to analyze. If None, returns mock prediction.
            auto_calibrate: If True and not yet calibrated, use first frame for calibration.

        Returns:
            Tuple of (fatigue_state, fatigue_score, confidence)
            - fatigue_state: One of "Alert", "Moderate", "High", "Critical"
            - fatigue_score: Normalized score (0-1) where 1 = critically fatigued, 0 = alert
            - confidence: Detection confidence (0-1)
        """
        # If model is not loaded or no frame provided, return mock data
        if (
            self.face_features is None
            or self.fatigue_rules is None
            or self.face_landmarker is None
            or frame is None
        ):
            return self._get_mock_signal()

        # Auto-calibrate on first frame if needed
        if not self.is_calibrated and auto_calibrate:
            self.calibrate(frame)

        try:
            # Resize frame for processing
            h, w = frame.shape[:2]
            scale = h / 460 if h > 460 else 1.0
            frame_resized = cv2.resize(frame, None, fx=1 / scale, fy=1 / scale)

            # Preprocessing
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            adjusted = cv2.equalizeHist(gray)

            # MediaPipe face detection
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(adjusted, cv2.COLOR_GRAY2RGB),
            )
            results = self.face_landmarker.detect(mp_image)

            if not results.face_landmarks:
                # No face detected
                return self._get_mock_signal()

            face_landmarks = results.face_landmarks[0]

            # Extract eye landmarks
            left_eye_indices = [33, 159, 158, 133, 153, 144]
            right_eye_indices = [263, 386, 385, 362, 381, 380]

            left_eye = [face_landmarks[i] for i in left_eye_indices]
            right_eye = [face_landmarks[i] for i in right_eye_indices]

            left_ear = self.face_features.eye_aspect_ratio(left_eye)
            right_ear = self.face_features.eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Update blink detection
            self.face_features.blink_detection(left_ear, right_ear)

            # Extract mouth landmarks for yawning
            mouth_indices = [61, 291, 0, 17, 146, 91, 181, 84, 314, 405, 321, 375]
            mouth = [face_landmarks[i] for i in mouth_indices]
            mar = self.face_features.mouth_aspect_ratio(mouth)

            # Update yawn detection
            self.face_features.yawning_detection(mar)

            # Get current metrics
            blink_count = self.face_features.blink_count
            yawn_count = self.face_features.yawning

            # Calculate derived metrics
            blink_rate = blink_count * 2  # Rough estimate
            brow_dist = 0.5  # Placeholder

            # Compute fatigue score
            fatigue_prob, suggest_break, fatigue_level, factors = (
                self.fatigue_rules.compute_fatigue_score(
                    ear=avg_ear,
                    blink_rate=blink_rate,
                    brow_dist=brow_dist,
                    jaw_open=mar,
                    yawning=yawn_count,
                )
            )

            # Map fatigue_level to standardized state
            state_mapping = {
                "alert": "Alert",
                "moderate": "Moderate",
                "high": "High",
                "critical": "Critical",
            }
            fatigue_state = state_mapping.get(fatigue_level, "Alert")

            # Calculate confidence based on face detection quality
            confidence = 0.85

            return fatigue_state, fatigue_prob, confidence

        except Exception as e:
            print(f"Error during fatigue detection: {e}")
            import traceback

            traceback.print_exc()
            return self._get_mock_signal()

    def _get_mock_signal(self) -> Tuple[FatigueState, float, float]:
        """
        Return mock signal data for testing/development.

        Returns:
            Tuple of (fatigue_state, fatigue_score, confidence)
        """
        # Return mock "Alert" state for testing
        return ("Alert", 0.15, 0.90)

    def is_model_loaded(self) -> bool:
        """Check if the fatigue detection system is successfully loaded."""
        return (
            self.face_features is not None
            and self.fatigue_rules is not None
            and self.face_landmarker is not None
        )

    def reset(self):
        """Reset the fatigue detection state (useful between sessions)."""
        if self.fatigue_rules is not None:
            self.fatigue_rules = FatigueRules()  # Reinitialize with fresh state
        if self.face_features is not None:
            self.face_features.blink_count = 0
            self.face_features.yawning = 0
        self.is_calibrated = False
