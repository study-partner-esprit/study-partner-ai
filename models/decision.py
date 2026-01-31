"""Decision model for agent outputs."""
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class Decision(BaseModel):
    """Model representing an agent's decision."""
    
    decision_id: Optional[str] = Field(
        None,
        description="Unique decision identifier"
    )
    agent_type: str = Field(
        ...,
        description="Type of agent making the decision"
    )
    decision_type: str = Field(
        ...,
        description="Type of decision made"
    )
    
    content: Dict[str, Any] = Field(
        ...,
        description="Decision content/payload"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    
    reasoning: Optional[str] = Field(
        None,
        description="Explanation of the decision reasoning"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Decision timestamp"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "decision_id": "dec_123abc",
                "agent_type": "planner",
                "decision_type": "plan",
                "content": {
                    "tasks": [
                        {
                            "title": "Review Python basics",
                            "duration": 30
                        }
                    ],
                    "total_duration": 30
                },
                "confidence": 0.85,
                "reasoning": "Based on user's beginner level and available time",
                "metadata": {
                    "model_version": "1.0",
                    "processing_time_ms": 150
                }
            }
        }
