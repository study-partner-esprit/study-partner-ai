"""
Pydantic schemas for the Reflection Agent.
"""

from pydantic import BaseModel, field_validator
from datetime import date
from typing import Optional, List


class DailyMetrics(BaseModel):
    """Daily study metrics input schema."""
    user_id: str
    date: date
    total_study_minutes: int
    avg_focus_score: float
    avg_fatigue_score: float
    xp_earned: int
    sessions_count: int

    @field_validator("avg_focus_score", "avg_fatigue_score")
    @classmethod
    def score_must_be_valid(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return v

    @field_validator("total_study_minutes", "xp_earned", "sessions_count")
    @classmethod
    def must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Value must be positive")
        return v


class WeeklySummary(BaseModel):
    """Weekly summary response schema."""
    user_id: str
    week_start: str
    week_end: str
    total_study_minutes: int
    total_xp_earned: int
    total_sessions: int
    days_studied: int
    avg_focus_score: float
    avg_fatigue_score: float
    best_focus_day: float
    worst_fatigue_day: float


class WeeklyHistoryItem(BaseModel):
    """Single week in history."""
    year: int
    week: int
    total_study_minutes: int
    total_xp_earned: int
    total_sessions: int
    days_studied: int
    avg_focus_score: float
    avg_fatigue_score: float


class TrendResponse(BaseModel):
    """Trend analysis response."""
    user_id: str
    weeks_analyzed: int
    latest_week: dict
    previous_week: dict
    progression_score: int
    trends: dict


class ReflectionResponse(BaseModel):
    """Generated reflection response."""
    user_id: str
    period: str
    progression_score: int
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    tips: List[str]
    trends_snapshot: dict
    generated_by: str
    created_at: str


class ErrorResponse(BaseModel):
    """Error response schema."""
    status: str = "error"
    detail: str