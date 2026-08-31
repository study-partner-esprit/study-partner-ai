"""Reflection, progress, review, and evaluation endpoints."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from deps import convert_objectid_to_str

router = APIRouter()


class ScheduleReviewRequest(BaseModel):
    user_id: str
    task_id: str
    task_title: str
    subject_tag: str = ""
    key_concepts: List[str] = []
    difficulty: str = "medium"


class RecordReviewResultRequest(BaseModel):
    user_id: str
    review_id: str
    quality_score: int  # 0-5


# ── Review / Spaced Repetition Endpoints ──────────────────────────────


@router.post("/api/ai/reviews/schedule")
async def schedule_review(request: ScheduleReviewRequest):
    """
    Schedule the first spaced repetition review for a completed task.
    """
    try:
        from agents.planner.memory.review_inserter import ReviewInserter

        inserter = ReviewInserter()
        result = inserter.schedule_review(
            user_id=request.user_id,
            task_id=request.task_id,
            task_title=request.task_title,
            subject_tag=request.subject_tag,
            key_concepts=request.key_concepts,
            difficulty=request.difficulty,
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to schedule review")
        return {"status": "ok", "review": convert_objectid_to_str(result)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/ai/reviews/record-result")
async def record_review_result(request: RecordReviewResultRequest):
    """
    Record the result of a review session and schedule the next review.

    Quality scores:
      0 = complete blackout
      1 = incorrect, recognized correct answer
      2 = incorrect, answer seemed easy to recall
      3 = correct with significant difficulty
      4 = correct after hesitation
      5 = perfect, instant recall
    """
    try:
        from agents.planner.memory.review_inserter import ReviewInserter

        inserter = ReviewInserter()
        result = inserter.record_review_result(
            user_id=request.user_id,
            review_id=request.review_id,
            quality_score=request.quality_score,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Review not found")
        return {"status": "ok", **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/ai/reviews/pending/{user_id}")
async def get_pending_reviews(user_id: str, limit: int = 20):
    """
    Get all pending and overdue reviews for a user.
    """
    try:
        from agents.planner.memory.review_inserter import ReviewInserter

        inserter = ReviewInserter()
        reviews = inserter.get_pending_reviews(user_id, limit=limit)
        return {"status": "ok", "reviews": reviews, "count": len(reviews)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/ai/reviews/stats/{user_id}")
async def get_review_stats(user_id: str):
    """
    Get review statistics for a user.
    """
    try:
        from agents.planner.memory.review_inserter import ReviewInserter

        inserter = ReviewInserter()
        stats = inserter.get_review_stats(user_id)
        return {"status": "ok", **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
