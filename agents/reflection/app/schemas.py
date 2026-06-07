from pydantic import BaseModel, field_validator
from datetime import date

class DailyMetrics(BaseModel):
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