"""Task model."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class Task(BaseModel):
    """Model representing a study task."""

    task_id: str = Field(..., description="Unique task identifier")
    user_id: str = Field(..., description="User this task belongs to")
    session_id: Optional[str] = Field(None, description="Associated session ID")

    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Detailed task description")

    priority: str = Field(
        default="medium", description="Task priority: low, medium, high"
    )
    difficulty: str = Field(default="medium", description="Task difficulty level")

    estimated_duration: int = Field(..., description="Estimated time in minutes")
    actual_duration: Optional[int] = Field(
        None, description="Actual time spent in minutes"
    )

    status: str = Field(
        default="pending",
        description="Task status: pending, in_progress, completed, skipped",
    )

    # Task type: 'study' for normal tasks, 'review' for spaced repetition reviews
    task_type: str = Field(default="study", description="Task type: study, review")

    # Spaced repetition metadata (only for review tasks)
    review_metadata: Optional["ReviewMetadata"] = Field(
        None, description="Spaced repetition metadata for review tasks"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")

    tags: list[str] = Field(
        default_factory=list, description="Task tags for categorization"
    )

    prerequisites: list[str] = Field(
        default_factory=list,
        description="List of task IDs that must be completed before this task",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_xyz789",
                "user_id": "user123",
                "session_id": "sess_abc123",
                "title": "Learn Python Variables",
                "description": "Understand variable declaration and types in Python",
                "priority": "high",
                "difficulty": "beginner",
                "estimated_duration": 30,
                "status": "pending",
                "task_type": "study",
                "tags": ["python", "basics", "variables"],
            }
        }


class ReviewMetadata(BaseModel):
    """Spaced repetition metadata for review tasks."""

    original_task_id: str = Field(
        ..., description="ID of the original task being reviewed"
    )
    original_task_title: str = Field(..., description="Title of the original task")
    subject_tag: str = Field(default="", description="Subject category")

    review_number: int = Field(
        default=1, description="Which review iteration this is (1st, 2nd, 3rd...)"
    )
    interval_days: int = Field(
        ..., description="Days since original completion or last review"
    )

    # Performance tracking
    ease_factor: float = Field(
        default=2.5,
        description="Ease factor (SM-2 algorithm style). Higher = easier recall. Range: 1.3 - 3.5",
    )
    quality_score: Optional[int] = Field(
        None,
        description="User self-reported recall quality (0-5). 0=complete blackout, 5=perfect recall",
    )

    key_concepts: List[str] = Field(
        default_factory=list,
        description="Key concepts to review from the original task",
    )

    next_review_date: Optional[datetime] = Field(
        None,
        description="Calculated next review date based on spaced repetition algorithm",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "original_task_id": "task_abc123",
                "original_task_title": "Learn Python Variables",
                "subject_tag": "python",
                "review_number": 2,
                "interval_days": 3,
                "ease_factor": 2.5,
                "quality_score": 4,
                "key_concepts": [
                    "variable declaration",
                    "type casting",
                    "naming conventions",
                ],
                "next_review_date": "2026-03-01T10:00:00",
            }
        }


# Update forward reference
Task.model_rebuild()
