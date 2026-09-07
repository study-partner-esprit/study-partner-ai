"""Signal snapshot model for capturing multi-modal signals.

This module defines the data structure for ML-based user state signals.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class SignalSnapshot(BaseModel):
    """
    Captures the current state of all ML-based user signals.

    This is the unified representation passed to the Coach agent
    to inform adaptive decision-making.
    """

    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Focus signal
    focus_state: Literal["Focused", "Drifting", "Lost"]
    focus_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score for focus state (0-1)"
    )

    # Fatigue signal
    fatigue_state: Literal["Alert", "Moderate", "High", "Critical"]
    fatigue_score: float = Field(
        ge=0.0, le=1.0, description="Fatigue probability (0=alert, 1=critical)"
    )

    # Confidence metadata
    focus_confidence: float = Field(
        ge=0.0, le=1.0, description="ML model confidence (0-1)"
    )
    fatigue_confidence: float = Field(
        ge=0.0, le=1.0, description="Fatigue detection confidence (0-1)"
    )

    # Short-term focus trend — slope of linear regression over a recent window
    # (e.g. 5 minutes).  Negative = focus declining, Positive = improving.
    # Populated by SignalRepository.compute_focus_trend().
    focus_trend: Optional[float] = Field(
        default=None,
        description="Linear trend slope of focus_score over the recent window. "
        "Negative = declining focus.",
    )

    # Emotion signal (COACH-15) — produced by EmotionAdapter, mirrors the
    # focus/fatigue state+confidence pattern. Optional so existing snapshots
    # without an emotion reading remain valid.
    affective_state: Optional[
        Literal["engaged", "frustrated", "stressed", "bored", "confident"]
    ] = Field(default=None, description="Mapped affective state from the FER model.")
    affective_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence of the mapped affective state (0-1)."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "timestamp": "2026-02-11T10:30:00",
                "focus_state": "Focused",
                "focus_score": 0.87,
                "focus_confidence": 0.92,
                "fatigue_state": "Alert",
                "fatigue_score": 0.15,
                "fatigue_confidence": 0.90,
                "focus_trend": -0.12,
            }
        }
