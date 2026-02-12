"""Focus adapter for loading and running the focus detection ML model.

This adapter wraps the local focus ML model and provides a clean interface
for getting focus signals without exposing model implementation details.
"""

import os
from typing import Literal, Tuple, Optional
from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available, focus adapter will use mock data")


class FocusAdapter:
    """
    Adapter for the focus detection ML model.
    
    Loads the trained Keras model and provides methods to get focus predictions.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the focus adapter.
        
        Args:
            model_path: Path to the trained model file. If None, uses default location.
        """
        if model_path is None:
            # Default path to the focus model
            base_path = Path(__file__).parent.parent.parent / "ML" / "focus" / "outputs" / "models"
            model_path = base_path / "focus_model.h5"
        
        self.model_path = str(model_path)
        self.model = None
        
        # Load model if TensorFlow is available and model exists
        if TF_AVAILABLE and os.path.exists(self.model_path):
            self._load_model()
        else:
            if not TF_AVAILABLE:
                print("Warning: TensorFlow not available")
            else:
                print(f"Warning: Focus model not found at {self.model_path}")
            print("Focus adapter will use mock data")
    
    def _load_model(self):
        """Load the trained Keras model."""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"Focus model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading focus model: {e}")
            self.model = None
    
    def get_focus_signal(
        self, 
        video_features: Optional[np.ndarray] = None
    ) -> Tuple[Literal["Focused", "Drifting", "Lost"], float, float]:
        """
        Get the current focus state based on video features.
        
        Args:
            video_features: Preprocessed video features (e.g., face landmarks, gaze vectors).
                           If None, returns a mock prediction.
        
        Returns:
            Tuple of (focus_state, focus_score, confidence)
            - focus_state: One of "Focused", "Drifting", "Lost"
            - focus_score: Normalized score (0-1) where 1 = fully focused
            - confidence: Model confidence in the prediction (0-1)
        """
        # If model is not loaded or no features provided, return mock data
        if self.model is None or video_features is None:
            return self._get_mock_signal()
        
        try:
            # Ensure features are in the right shape for Keras model
            if video_features.ndim == 1:
                video_features = np.expand_dims(video_features, axis=0)
            
            # Get model prediction
            predictions = self.model.predict(video_features, verbose=0)
            
            # Assuming output is [batch, 3] for three classes (Focused, Drifting, Lost)
            if predictions.shape[-1] == 3:
                # Classification with probabilities
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                # Map class index to state
                states = ["Focused", "Drifting", "Lost"]
                focus_state = states[predicted_class]
                
                # Calculate focus score (inverse of distraction)
                # Focused = 1.0, Drifting = 0.5, Lost = 0.0
                focus_score = 1.0 - (predicted_class / 2.0)
            else:
                # Regression output (single value)
                focus_score = float(predictions[0][0])
                confidence = 0.8  # Default confidence for regression
                
                # Map score to state
                if focus_score > 0.7:
                    focus_state = "Focused"
                elif focus_score > 0.3:
                    focus_state = "Drifting"
                else:
                    focus_state = "Lost"
            
            return focus_state, focus_score, confidence
        
        except Exception as e:
            print(f"Error during focus prediction: {e}")
            return self._get_mock_signal()
    
    def _get_mock_signal(self) -> Tuple[Literal["Focused", "Drifting", "Lost"], float, float]:
        """
        Return mock signal data for testing/development.
        
        Returns:
            Tuple of (focus_state, focus_score, confidence)
        """
        # Return mock "Focused" state for testing
        return ("Focused", 0.85, 0.90)
    
    def is_model_loaded(self) -> bool:
        """Check if the ML model is successfully loaded."""
        return self.model is not None
