"""Task graph models for planning using Pydantic."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class AtomicTask(BaseModel):
    """
    Represents an atomic task in the task graph.
    
    Each task is a small, focused learning activity that can be completed
    in a single session (5-45 minutes).
    """
    id: str = Field(..., description="Unique identifier for the task")
    title: str = Field(..., description="Short, descriptive title")
    description: str = Field(..., description="Detailed description of what to learn/do")
    estimated_minutes: int = Field(..., ge=5, le=45, description="Estimated time in minutes (5-45)")
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0, description="Difficulty level (0.0-1.0)")
    prerequisites: List[str] = Field(default_factory=list, description="List of prerequisite task IDs")
    is_review: bool = Field(default=False, description="Whether this is a review task")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "task-001",
                "title": "Python Basics",
                "description": "Learn Python syntax and basic data types",
                "estimated_minutes": 30,
                "difficulty": 0.3,
                "prerequisites": [],
                "is_review": False
            }
        }


class PlannerInput(BaseModel):
    """
    Input for the planner agent.
    
    Contains all information needed to generate a personalized study plan.
    """
    goal: str = Field(..., description="Main learning goal")
    deadline_iso: str = Field(..., description="Deadline in ISO 8601 format")
    available_minutes: int = Field(..., gt=0, description="Total available time in minutes")
    user_id: str = Field(..., description="User identifier")
    retrieved_concepts: Optional[List[str]] = Field(default=None, description="RAG-retrieved relevant concepts")
    course_documents: Optional[List[str]] = Field(default=None, description="Course materials/documents")
    tokenization_settings: Optional[Dict[str, Any]] = Field(default=None, description="Settings for tokenization")

    @field_validator('deadline_iso')
    def validate_deadline(cls, v):
        """Validate that deadline_iso is a valid ISO 8601 datetime."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 datetime: {v}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "goal": "Learn Python for Machine Learning",
                "deadline_iso": "2026-02-09T00:00:00",
                "available_minutes": 300,
                "user_id": "user123",
                "retrieved_concepts": ["NumPy", "Pandas", "Scikit-learn"],
                "course_documents": [],
                "tokenization_settings": {}
            }
        }


class TaskGraph(BaseModel):
    """
    Represents a graph of tasks for achieving a learning goal.
    
    Contains the sequence of atomic tasks needed to complete the goal.
    """
    goal: str = Field(..., description="The learning goal")
    tasks: List[AtomicTask] = Field(..., description="List of atomic tasks")
    total_estimated_minutes: int = Field(default=0, description="Total time for all tasks")

    def model_post_init(self, __context):
        """Calculate total estimated minutes after initialization."""
        if self.tasks:
            self.total_estimated_minutes = sum(task.estimated_minutes for task in self.tasks)

    class Config:
        json_schema_extra = {
            "example": {
                "goal": "Learn Python basics",
                "tasks": [],
                "total_estimated_minutes": 0
            }
        }


class PlannerOutput(BaseModel):
    """
    Output from the planner agent.
    
    Contains the generated task graph and any warnings or clarification requests.
    """
    task_graph: TaskGraph = Field(..., description="Generated task graph")
    warning: Optional[str] = Field(default=None, description="Warning message if any issues")
    clarification_required: bool = Field(default=False, description="Whether user input needs clarification")

    class Config:
        json_schema_extra = {
            "example": {
                "task_graph": {
                    "goal": "Learn Python basics",
                    "tasks": [],
                    "total_estimated_minutes": 0
                },
                "warning": None,
                "clarification_required": False
            }
        }
