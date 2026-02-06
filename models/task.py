"""Task model."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Task(BaseModel):
    """Model representing a study task."""
    
    task_id: str = Field(..., description="Unique task identifier")
    user_id: str = Field(..., description="User this task belongs to")
    session_id: Optional[str] = Field(None, description="Associated session ID")
    
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Detailed task description")
    
    priority: str = Field(
        default="medium",
        description="Task priority: low, medium, high"
    )
    difficulty: str = Field(
        default="medium",
        description="Task difficulty level"
    )
    
    estimated_duration: int = Field(
        ...,
        description="Estimated time in minutes"
    )
    actual_duration: Optional[int] = Field(
        None,
        description="Actual time spent in minutes"
    )
    
    status: str = Field(
        default="pending",
        description="Task status: pending, in_progress, completed, skipped"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    started_at: Optional[datetime] = Field(
        None,
        description="Start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Completion timestamp"
    )
    
    tags: list[str] = Field(
        default_factory=list,
        description="Task tags for categorization"
    )
    
    prerequisites: list[str] = Field(
        default_factory=list,
        description="List of task IDs that must be completed before this task"
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
                "tags": ["python", "basics", "variables"]
            }
        }
