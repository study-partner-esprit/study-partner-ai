"""Session models and request/response schemas."""
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class SessionRequest(BaseModel):
    """Request model for session interactions."""
    
    user_id: str = Field(..., description="User identifier")
    signal_type: str = Field(..., description="Type of signal being sent")
    session_id: Optional[str] = Field(None, description="Existing session ID")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context data"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Request timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "signal_type": "start_session",
                "context": {
                    "goals": ["Learn Python basics", "Build a project"],
                    "available_time": 120,
                    "difficulty": "beginner"
                }
            }
        }


class SessionResponse(BaseModel):
    """Response model for session interactions."""
    
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    agent_type: str = Field(..., description="Agent that processed the request")
    decision_type: str = Field(..., description="Type of decision made")
    content: Dict[str, Any] = Field(..., description="Response content")
    confidence: float = Field(..., description="Confidence score (0-1)")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "user_id": "user123",
                "agent_type": "planner",
                "decision_type": "plan",
                "content": {
                    "tasks": [],
                    "estimated_duration": 120
                },
                "confidence": 0.85
            }
        }


class SessionState(BaseModel):
    """Model representing current session state."""
    
    session_id: str
    user_id: str
    state: str  # active, paused, completed
    started_at: datetime
    ended_at: Optional[datetime] = None
    current_task_id: Optional[str] = None
    completed_tasks: int = 0
    total_time: int = 0  # in minutes
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "user_id": "user123",
                "state": "active",
                "started_at": "2026-01-30T10:00:00Z",
                "completed_tasks": 3,
                "total_time": 45
            }
        }
