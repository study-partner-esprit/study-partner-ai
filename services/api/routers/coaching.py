"""Coaching and conversation endpoints."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from deps import get_ai_orchestrator, get_schedule_orchestrator

router = APIRouter()


class CoachRequest(BaseModel):
    user_id: str
    ignored_count: int = 0
    do_not_disturb: bool = False
    # Live signal overrides from frontend webcam pipeline
    focus_score: Optional[float] = None
    focus_state: Optional[str] = None
    fatigue_score: Optional[float] = None
    fatigue_state: Optional[str] = None


@router.post("/api/ai/coach/decision")
async def get_coach_decision(
    request: CoachRequest,
    x_trace_id: Optional[str] = Header(None, alias="x-trace-id"),
):
    """
    Get real-time coaching decision based on current context.

    Args:
        request: CoachRequest with user ID and context
        x_trace_id: Optional request trace ID forwarded from the API gateway

    Returns:
        CoachAction with decision and optional schedule changes
    """
    try:
        # Run coach through orchestrator — propagate trace_id
        coach_action = get_ai_orchestrator().run_coach(
            user_id=request.user_id,
            current_time=datetime.now(),
            ignored_count=request.ignored_count,
            do_not_disturb=request.do_not_disturb,
            trace_id=x_trace_id,
            live_focus_score=request.focus_score,
            live_focus_state=request.focus_state,
            live_fatigue_score=request.fatigue_score,
            live_fatigue_state=request.fatigue_state,
        )

        # If coach suggests schedule changes, implement them
        if coach_action.schedule_changes:
            schedule_result = get_schedule_orchestrator().process_coach_action(
                coach_action=coach_action,
                user_id=request.user_id,
                current_time=datetime.now(),
            )
            return {
                "coach_action": coach_action.model_dump(),
                "schedule_update": schedule_result,
            }

        return {"coach_action": coach_action.model_dump(), "schedule_update": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coach decision failed: {str(e)}")


@router.get("/api/ai/coach/history/{user_id}")
async def get_coach_history(user_id: str, limit: int = 20):
    """Get recent coaching action history for a user."""
    try:
        from agents.coach.services.coach_history_repository import (
            CoachHistoryRepository,
        )

        repo = CoachHistoryRepository()
        history = repo.get_recent_actions(user_id, limit=limit)
        return {"history": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch history: {str(e)}"
        )
