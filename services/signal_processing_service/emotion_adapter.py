"""Emotion adapter for loading and running the facial emotion recognition
model (COACH-15).

Mirrors FocusAdapter / FatigueAdapter: wraps the local FER ML model and
MediaPipe face detection, and provides a clean `get_emotion_signal()`
interface without exposing model implementation details to the rest of the
signal processing service.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from services.signal_processing_service.emotion_mapping import (
    FER_EMOTIONS as EMOTIONS,
    map_probabilities_to_affective_state,
)

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from tensorflow.keras import layers, models, regularizers

    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    print(f"Warning: emotion adapter dependencies not available ({e}), using mock data")

TEMPERATURE = 2.093674898147583


class EmotionAdapter:
    """
    Adapter for the facial emotion recognition model (7-class FER CNN).

    Loads the trained TensorFlow checkpoint + MediaPipe face detector, runs
    the same pipeline validated in webcam_emotion.py / stress_test_validation.py
    (grayscale 48x48, /255.0 normalization, temperature-calibrated softmax),
    then maps the 7 probabilities onto the Coach's affective_state vocabulary.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        face_model_path: Optional[str] = None,
    ):
        """
        Args:
            weights_path: Path to the Keras 3 `.weights.h5` file, converted
                once from the original TensorFlow native checkpoint (`.index`
                + `.data-00000-of-00001`) — Keras 3 (bundled with
                TensorFlow>=2.16) no longer accepts that native checkpoint
                format for `load_weights()`. See
                docs/emotion_checkpoint_conversion.md for the one-time
                conversion script (run once, in the original Keras 2
                environment where the checkpoint was produced).
                If None, uses the default location under ML/emotion/.
            face_model_path: Path to the MediaPipe face detector .tflite
                model. If None, uses the default location under ML/emotion/.
        """
        if weights_path is None:
            base_path = (
                Path(__file__).parent.parent.parent
                / "ML"
                / "emotion"
                / "outputs"
                / "models"
            )
            weights_path = base_path / "emotion_model_cnn_from_scratch_v3.weights.h5"
        if face_model_path is None:
            base_path = Path(__file__).parent.parent.parent / "ML" / "emotion"
            face_model_path = base_path / "detector.tflite"

        self.weights_path = str(weights_path)
        self.face_model_path = str(face_model_path)
        self.model = None
        self.detector = None

        if DEPS_AVAILABLE:
            self._load_model()
            self._load_face_detector()
        else:
            print("Emotion adapter will use mock data")

    def _load_model(self):
        """Reconstruct the architecture and load the available model weights."""
        try:
            if not os.path.exists(self.weights_path):
                print(f"Warning: emotion model weights not found at {self.weights_path}")
                return
            self.model = self._build_model()
            try:
                self.model.load_weights(self.weights_path)
                loaded_from = self.weights_path
            except Exception as h5_error:
                npz_path = Path(self.weights_path).parent / "emotion_model_weights_arrays.npz"
                if not npz_path.exists():
                    raise h5_error
                weights = np.load(npz_path)
                self.model.set_weights([weights[key] for key in weights.files])
                loaded_from = str(npz_path)
                print(f"Warning: H5 weights incompatible; loaded NPZ weights instead ({h5_error})")
            print(f"Emotion model loaded successfully from {loaded_from}")
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            self.model = None

    @staticmethod
    def _build_model():
        """4 blocks of 2xConv2D-BatchNorm, MaxPooling2D, increasing dropout,
        then GlobalAveragePooling2D -> Dense(128) -> Dense(7, softmax).
        Must match training-time architecture exactly or load_weights()
        will map weights onto the wrong shapes."""
        inputs = layers.Input(shape=(48, 48, 1), name="input_image")
        x = inputs
        for filters, dropout in [(32, 0.25), (64, 0.30), (128, 0.35)]:
            x = layers.Conv2D(
                filters, 3, padding="same", activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(
                filters, 3, padding="same", activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(dropout)(x)

        x = layers.Conv2D(
            256, 3, padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.40)(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(7, activation="softmax", name="emotion_output")(x)

        return models.Model(inputs, outputs, name="cnn_from_scratch")

    def _load_face_detector(self):
        try:
            if not os.path.exists(self.face_model_path):
                print(f"Warning: face detector model not found at {self.face_model_path}")
                return
            base_options = mp_python.BaseOptions(model_asset_path=self.face_model_path)
            options = vision.FaceDetectorOptions(base_options=base_options)
            self.detector = vision.FaceDetector.create_from_options(options)
        except Exception as e:
            print(f"Error loading face detector: {e}")
            self.detector = None

    def get_emotion_signal(
        self, frame: Optional[np.ndarray] = None
    ) -> Tuple[str, float]:
        """
        Get the current affective state based on a video frame.

        Args:
            frame: Video frame (BGR format) to analyze. If None, or if the
                   model/detector are not loaded, returns a mock prediction.

        Returns:
            Tuple of (affective_state, confidence)
            - affective_state: one of "engaged", "frustrated", "stressed",
              "bored", "confident" — "bored" is never produced by a single
              frame (see emotion_mapping.map_probabilities_to_affective_state)
            - confidence: mapped-group confidence (0-1)
        """
        if self.model is None or self.detector is None or frame is None:
            return self._get_mock_signal()

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.detector.detect(mp_image)

            if not result.detections:
                return self._get_mock_signal()

            # Largest detected face, consistent with webcam_emotion.py
            best = max(
                result.detections,
                key=lambda d: d.bounding_box.width * d.bounding_box.height,
            )
            bbox = best.bounding_box
            h, w = frame.shape[:2]
            x1, y1 = max(bbox.origin_x, 0), max(bbox.origin_y, 0)
            x2, y2 = min(x1 + bbox.width, w), min(y1 + bbox.height, h)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0 or min(face_crop.shape[:2]) < 15:
                return self._get_mock_signal()

            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized.astype("float32") / 255.0
            x = normalized.reshape(1, 48, 48, 1)

            raw_probs = self.model.predict(x, verbose=0)[0]
            calibrated = self._apply_temperature(raw_probs)

            probabilities = {name: float(calibrated[i]) for i, name in enumerate(EMOTIONS)}
            return map_probabilities_to_affective_state(probabilities)

        except Exception as e:
            print(f"Error during emotion prediction: {e}")
            return self._get_mock_signal()

    @staticmethod
    def _apply_temperature(
        softmax_probs: np.ndarray, T: float = TEMPERATURE, eps: float = 1e-9
    ) -> np.ndarray:
        logits = np.log(np.clip(softmax_probs, eps, 1.0))
        scaled = np.exp(logits / T)
        return scaled / scaled.sum()

    def _get_mock_signal(self) -> Tuple[str, float]:
        """
        Return mock signal data for testing/development.

        Unlike FocusAdapter/FatigueAdapter (which mock a high-confidence
        plausible state), this returns confidence 0.0 on purpose: it
        guarantees the confidence-threshold fallback in
        SignalProcessingService always kicks in when the emotion model is
        genuinely unavailable, instead of confidently asserting "engaged".
        """
        return ("engaged", 0.0)

    def is_model_loaded(self) -> bool:
        """Check if the ML model and face detector are successfully loaded."""
        return self.model is not None and self.detector is not None
