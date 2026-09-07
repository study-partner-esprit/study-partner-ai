"""Signal service façade for orchestrating ML model adapters.

This service coordinates the collection of signals from multiple ML models,
normalizes their outputs, applies confidence thresholds, and persists the results.
"""

from datetime import datetime
from typing import Optional

# Lazy imports to avoid startup crashes
# from services.signal_processing_service.focus_adapter import FocusAdapter
# from services.signal_processing_service.fatigue_adapter import FatigueAdapter
from services.signal_processing_service.repository import SignalRepository
from services.signal_processing_service.signal_snapshot import SignalSnapshot


class SignalProcessingService:
    """
    Façade for the signal processing subsystem.

    Coordinates multiple ML model adapters to produce a unified SignalSnapshot.
    """

    # Confidence thresholds for accepting predictions
    MIN_CONFIDENCE_THRESHOLD = 0.6

    def __init__(self):
        """Initialize all ML adapters and the repository."""
        self.focus_adapter = None
        self.fatigue_adapter = None
        self._initialized = False

        try:
            from services.signal_processing_service.fatigue_adapter import (
                FatigueAdapter,
            )
            from services.signal_processing_service.focus_adapter import FocusAdapter

            self.focus_adapter = FocusAdapter()
            self.fatigue_adapter = FatigueAdapter()
            self._initialized = True
            print("✓ Signal processing service initialized successfully")
        except Exception as e:
            print(f"⚠ Signal processing service initialization failed: {e}")
            self._initialized = False

        # COACH-15: emotion adapter is initialized separately from
        # focus/fatigue on purpose. A missing/broken emotion model must
        # degrade gracefully (affective_state falls back downstream) and
        # must NOT disable focus/fatigue signal processing.
        self.emotion_adapter = None
        self._emotion_initialized = False
        try:
            from services.signal_processing_service.emotion_adapter import (
                EmotionAdapter,
            )

            self.emotion_adapter = EmotionAdapter()
            self._emotion_initialized = True
            print("✓ Emotion adapter initialized successfully")
        except Exception as e:
            print(f"⚠ Emotion adapter initialization failed: {e}")
            self._emotion_initialized = False

        self.repository = SignalRepository()

    def is_ready(self) -> bool:
        """Check if the signal processing service is ready to process signals."""
        return getattr(self, "_initialized", False)

    def is_emotion_ready(self) -> bool:
        """Check if the emotion adapter is ready to process signals."""
        return getattr(self, "_emotion_initialized", False)

    def get_current_signal_snapshot(
        self,
        user_id: str,
        video_features: Optional[any] = None,
        video_frame: Optional[any] = None,
    ) -> SignalSnapshot:
        """
        Collect signals from all ML models and produce a unified snapshot.

        This is the main entry point for obtaining the current user state.

        Args:
            user_id: The user's unique identifier
            video_features: Optional preprocessed video features for focus detection
            video_frame: Optional video frame (BGR numpy array) for fatigue detection

        Returns:
            A SignalSnapshot containing all current signals
        """
        if not self.is_ready():
            # Return a default snapshot if models aren't loaded
            return SignalSnapshot(
                user_id=user_id,
                timestamp=datetime.now(),
                focus_state="unknown",
                focus_score=0.5,
                focus_confidence=0.0,
                fatigue_state="unknown",
                fatigue_score=0.5,
                fatigue_confidence=0.0,
                overall_state="unknown",
                dnd=False,
            )

        # Collect focus signal
        focus_state, focus_score, focus_confidence = (
            self.focus_adapter.get_focus_signal(video_features=video_features)
        )

        # Collect fatigue signal
        fatigue_state, fatigue_score, fatigue_confidence = (
            self.fatigue_adapter.get_fatigue_signal(frame=video_frame)
        )

        # Collect emotion signal (COACH-15). Reuses the same raw video_frame
        # as fatigue detection — the FER model runs its own MediaPipe face
        # detector on it, independent of the fatigue landmarker. Missing/
        # unready adapter degrades to a neutral, zero-confidence reading
        # rather than raising, so focus/fatigue processing is unaffected.
        if self.is_emotion_ready():
            affective_state, affective_confidence = (
                self.emotion_adapter.get_emotion_signal(frame=video_frame)
            )
        else:
            affective_state, affective_confidence = "engaged", 0.0

        # Apply confidence threshold (same policy extended to the emotion
        # signal: any low-confidence reading — focus, fatigue, or emotion —
        # triggers the recent-snapshot fallback).
        if (
            focus_confidence < self.MIN_CONFIDENCE_THRESHOLD
            or fatigue_confidence < self.MIN_CONFIDENCE_THRESHOLD
            or affective_confidence < self.MIN_CONFIDENCE_THRESHOLD
        ):
            # Low confidence - check if we have a recent reliable signal
            recent_snapshot = self.repository.get_latest_signal_snapshot(user_id)
            if recent_snapshot is not None:
                # Use the recent signal if it's not too old (within 5 minutes)
                age_seconds = (
                    datetime.now() - recent_snapshot.timestamp
                ).total_seconds()
                if age_seconds < 300:  # 5 minutes
                    print(
                        f"Using recent signal due to low confidence "
                        f"(focus: {focus_confidence:.2f}, fatigue: {fatigue_confidence:.2f}, "
                        f"affective: {affective_confidence:.2f})"
                    )
                    return recent_snapshot

            # No recent signal - use the low-confidence prediction anyway
            print(
                f"Warning: Low confidence for signals "
                f"(focus: {focus_confidence:.2f}, fatigue: {fatigue_confidence:.2f}, "
                f"affective: {affective_confidence:.2f})"
            )

        # Create snapshot
        snapshot = SignalSnapshot(
            user_id=user_id,
            timestamp=datetime.now(),
            focus_state=focus_state,
            focus_score=focus_score,
            focus_confidence=focus_confidence,
            fatigue_state=fatigue_state,
            fatigue_score=fatigue_score,
            fatigue_confidence=fatigue_confidence,
            affective_state=affective_state,
            affective_confidence=affective_confidence,
        )

        # Persist the snapshot
        snapshot_id = self.repository.save_signal_snapshot(snapshot)
        print(f"Signal snapshot saved with ID: {snapshot_id}")

        return snapshot

    def get_latest_snapshot(self, user_id: str) -> Optional[SignalSnapshot]:
        """
        Retrieve the most recent signal snapshot for a user.

        Args:
            user_id: The user's unique identifier

        Returns:
            The latest SignalSnapshot, or None if none exists
        """
        return self.repository.get_latest_signal_snapshot(user_id)
