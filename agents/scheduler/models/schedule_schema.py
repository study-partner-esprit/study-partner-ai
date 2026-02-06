from datetime import datetime
from typing import List

from pydantic import BaseModel, Field, ConfigDict


class ScheduledSession(BaseModel):
    """
    One concrete scheduled study session.
    """

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )

    task_id: str = Field(..., description="ID of the atomic study task")
    start_datetime: datetime
    end_datetime: datetime

    # break after this session (minutes)
    break_after_minutes: int = Field(
        default=5,
        ge=0,
        description="Break duration after this session in minutes",
    )

    # heuristic score used internally by the scheduler
    slot_score: float = Field(
        ...,
        ge=0.0,
        description="Heuristic score of the selected time slot",
    )
    
    scheduled: bool = Field(
        default=True,
        description="Whether this session was successfully scheduled",
    )


class StudyPlan(BaseModel):
    """
    Final scheduling result for a planning request.
    """

    sessions: List[ScheduledSession]

    total_minutes: int = Field(
        ...,
        ge=0,
        description="Total planned study time in minutes",
    )

    fallback_used: bool = Field(
        default=False,
        description="True if the scheduler used the fallback strategy",
    )
    
    skipped_tasks: List[str] = Field(
        default_factory=list,
        description="Task IDs that could not be scheduled (missing prerequisites, no space, etc.)",
    )
    
    span_days: int = Field(
        default=1,
        ge=1,
        description="Number of calendar days the plan spans",
    )
