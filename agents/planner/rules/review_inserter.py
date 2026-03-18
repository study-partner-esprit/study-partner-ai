"""
Review task inserter for the planner rules pipeline.

Integrates with the ReviewInserter from planner memory to inject
spaced repetition review tasks into generated study plans.
"""

from __future__ import annotations

from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def insert_review_tasks(
    tasks, user_id: str = "", pending_reviews: Optional[List] = None
):
    """
    Insert pending review tasks into the existing task list.

    Reviews are interleaved with study tasks, prioritized by urgency:
    - Overdue reviews are placed at the beginning
    - Scheduled reviews are distributed among study tasks

    Args:
        tasks:           List of AtomicTask objects from the planner.
        user_id:         User identifier for fetching pending reviews.
        pending_reviews: Pre-fetched pending reviews (optional).

    Returns:
        Updated list of tasks with review tasks interleaved.
    """
    if not user_id and not pending_reviews:
        return tasks

    reviews = pending_reviews or []

    # Fetch pending reviews if not provided
    if not reviews and user_id:
        try:
            from agents.planner.memory.review_inserter import ReviewInserter

            inserter = ReviewInserter()
            reviews = inserter.get_pending_reviews(user_id, limit=10)
        except Exception as exc:
            logger.warning("review_fetch_failed", extra={"error": str(exc)})
            return tasks

    if not reviews:
        return tasks

    # Convert reviews to AtomicTask-like format
    review_tasks = []
    for review in reviews:
        try:
            from agents.planner.models.task_graph import AtomicTask

            review_task = AtomicTask(
                id=review.get("review_id", f"review-{len(review_tasks)}"),
                title=f"📝 Review: {review.get('original_task_title', 'Unknown')}",
                description=(
                    f"Spaced repetition review #{review.get('review_number', 1)} — "
                    f"Key concepts: {', '.join(review.get('key_concepts', ['general review']))}"
                ),
                estimated_minutes=min(
                    15, 5 + review.get("review_number", 1) * 2
                ),  # 7-15 min based on review number
                difficulty=0.3,  # Reviews are generally easier than new material
                prerequisites=[],
                is_review=True,
            )
            review_tasks.append((review_task, review.get("status", "scheduled")))
        except Exception as exc:
            logger.warning(
                "review_task_creation_failed",
                extra={"error": str(exc), "review_id": review.get("review_id")},
            )

    if not review_tasks:
        return tasks

    # Separate overdue and scheduled reviews
    overdue = [rt for rt, status in review_tasks if status == "overdue"]
    scheduled = [rt for rt, status in review_tasks if status != "overdue"]

    # Build final task list: overdue first, then interleave scheduled
    result = list(overdue)  # Overdue reviews go first

    if scheduled and tasks:
        # Distribute scheduled reviews evenly among study tasks
        interval = max(1, len(tasks) // (len(scheduled) + 1))
        scheduled_iter = iter(scheduled)

        for i, task in enumerate(tasks):
            result.append(task)
            if (i + 1) % interval == 0:
                try:
                    result.append(next(scheduled_iter))
                except StopIteration:
                    pass

        # Append any remaining scheduled reviews
        for remaining in scheduled_iter:
            result.append(remaining)
    else:
        result.extend(tasks)
        result.extend([rt for rt, _ in review_tasks if rt not in overdue])

    logger.info(
        "review_tasks_inserted",
        extra={
            "original_count": len(tasks),
            "overdue_count": len(overdue),
            "scheduled_count": len(scheduled),
            "total_count": len(result),
        },
    )

    return result
