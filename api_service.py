"""FastAPI HTTP service for the AI system.

This service exposes HTTP endpoints for the Express backend to interact with
the Python AI system (orchestrator, coach, ML models).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import uvicorn

from services.ai_orchestrator.orchestrator import AIOrchestrator
from agents.coach.models.schemas import CoachAction


# Initialize FastAPI app
app = FastAPI(
    title="Study Partner AI Service",
    description="Python AI service for ML model integration and agent orchestration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = AIOrchestrator()


# Request/Response models
class CoachRunRequest(BaseModel):
    """Request body for running the coach agent."""
    user_id: str
    ignored_count: Optional[int] = 0
    do_not_disturb: Optional[bool] = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "ignored_count": 0,
                "do_not_disturb": False
            }
        }


class CoachRunResponse(BaseModel):
    """Response from running the coach agent."""
    action_type: str
    message: Optional[str]
    reasoning: str
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "silence",
                "message": None,
                "reasoning": "User is deeply focused (ML confidence: 0.92). Never interrupt productive flow state.",
                "timestamp": "2026-02-11T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    timestamp: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="study-partner-ai",
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/coach/run", response_model=CoachRunResponse)
async def run_coach(request: CoachRunRequest):
    """
    Execute the Coach agent with full context.
    
    This endpoint:
    1. Fetches the latest scheduled tasks for the user from MongoDB
    2. Fetches the latest SignalSnapshot (ML signals) for the user
    3. Calls the AI orchestrator to execute the coach agent
    4. Returns the coach's decision/action
    
    Args:
        request: CoachRunRequest with user_id and optional parameters
    
    Returns:
        CoachRunResponse with the coach's decision
    
    Raises:
        HTTPException: If execution fails
    """
    try:
        # Execute the coach through the orchestrator
        coach_action: CoachAction = orchestrator.run_coach(
            user_id=request.user_id,
            current_time=datetime.now(),
            ignored_count=request.ignored_count,
            do_not_disturb=request.do_not_disturb
        )
        
        # Convert to response format
        return CoachRunResponse(
            action_type=coach_action.action_type,
            message=coach_action.message,
            reasoning=coach_action.reasoning,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        # Log the error and return 500
        print(f"Error executing coach: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute coach agent: {str(e)}"
        )


@app.get("/api/signals/{user_id}")
async def get_user_signals(user_id: str):
    """
    Get the latest signal snapshot for a user.
    
    Args:
        user_id: The user's unique identifier
    
    Returns:
        The latest SignalSnapshot or 404 if none exists
    """
    try:
        from services.signal_processing_service.service import SignalProcessingService
        
        signal_service = SignalProcessingService()
        snapshot = signal_service.get_latest_snapshot(user_id)
        
        if snapshot is None:
            raise HTTPException(
                status_code=404,
                detail=f"No signal snapshot found for user {user_id}"
            )
        
        return snapshot.model_dump()
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching signals: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch signals: {str(e)}"
        )


@app.post("/api/signals/{user_id}/collect")
async def collect_user_signals(user_id: str):
    """
    Collect fresh ML signals for a user and persist them.
    
    Args:
        user_id: The user's unique identifier
    
    Returns:
        The newly created SignalSnapshot
    """
    try:
        from services.signal_processing_service.service import SignalProcessingService
        
        signal_service = SignalProcessingService()
        snapshot = signal_service.get_current_signal_snapshot(user_id)
        
        return snapshot.model_dump()
    
    except Exception as e:
        print(f"Error collecting signals: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to collect signals: {str(e)}"
        )


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
