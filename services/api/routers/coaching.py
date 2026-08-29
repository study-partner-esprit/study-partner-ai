"""Coaching and conversation endpoints.

F03 / COACH-01: coaching decisions no longer run through direct HTTP. The
`study.coach.nudge` job bus path (CoachWorker) replaced the synchronous
`POST /api/ai/coach/decision` route. Only the passive history read remains.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException

router = APIRouter()


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