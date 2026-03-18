"""
ReviewInserter — Spaced Repetition Review Task Generator.

Implements a modified SM-2 algorithm to generate review tasks at
optimal intervals based on the Ebbinghaus forgetting curve.

Review intervals follow modified Fibonacci-like spacing:
  Review 1: 1 day after completion
  Review 2: 3 days after Review 1
  Review 3: 5 days after Review 2
  Review 4: 8 days after Review 3
  Review 5: 13 days after Review 4
  Review 6+: interval * ease_factor

The ease factor adjusts based on user recall quality:
  - quality 0-1 (blackout/fail): ease_factor -= 0.3, reset to Review 1
  - quality 2 (hard):            ease_factor -= 0.15
  - quality 3 (okay):            ease_factor unchanged
  - quality 4 (good):            ease_factor += 0.1
  - quality 5 (perfect):         ease_factor += 0.15

Minimum ease factor: 1.3
Maximum ease factor: 3.5
Default ease factor: 2.5

MongoDB Collection: review_schedule
Schema:
  {
    user_id:            str,
    original_task_id:   str,
    original_task_title: str,
    subject_tag:        str,
    key_concepts:       List[str],
    review_number:      int,       # current review iteration
    ease_factor:        float,     # SM-2 ease factor
    interval_days:      int,       # days until next review
    next_review_date:   datetime,  # when to schedule next review
    last_review_date:   datetime,  # when last reviewed
    quality_history:    List[int], # past quality scores
    status:             str,       # 'pending' | 'scheduled' | 'completed' | 'overdue'
    created_at:         datetime,
  }
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

REVIEW_COLLECTION = "review_schedule"

# Modified Fibonacci intervals (days) for the first 6 reviews
BASE_INTERVALS = [1, 3, 5, 8, 13, 21]

# SM-2 ease factor bounds
MIN_EASE = 1.3
MAX_EASE = 3.5
DEFAULT_EASE = 2.5


class ReviewInserter:
    """
    Generates and manages spaced repetition review tasks.

    Uses a modified SM-2 algorithm with Fibonacci-like base intervals
    that adapt based on user recall quality.
    """

    def __init__(self) -> None:
        self._db = None
        try:
            from services.database import get_db

            self._db = get_db()
        except Exception as exc:
            logger.warning("review_inserter_no_db", extra={"error": str(exc)})

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def schedule_review(
        self,
        user_id: str,
        task_id: str,
        task_title: str,
        subject_tag: str = "",
        key_concepts: Optional[List[str]] = None,
        difficulty: str = "medium",
    ) -> Optional[Dict]:
        """
        Schedule the first review for a completed task.

        Called when a user completes a study task. Creates the initial
        review entry with a 1-day interval.

        Args:
            user_id:       Unique user identifier.
            task_id:       The completed task ID.
            task_title:    Title of the completed task.
            subject_tag:   Subject category (e.g. "math", "python").
            key_concepts:  List of key concepts from the task.
            difficulty:    Task difficulty: easy, medium, hard.

        Returns:
            Dict with the scheduled review info, or None if DB unavailable.
        """
        if self._db is None:
            logger.warning("review_inserter_no_db_skip")
            return None

        if not user_id or not task_id:
            logger.warning("review_inserter_missing_params")
            return None

        now = datetime.now(timezone.utc)

        # Adjust initial ease factor based on difficulty
        initial_ease = DEFAULT_EASE
        if difficulty == "easy":
            initial_ease = 2.8
        elif difficulty == "hard":
            initial_ease = 2.2

        first_interval = BASE_INTERVALS[0]  # 1 day
        next_review = now + timedelta(days=first_interval)

        review_doc = {
            "review_id": f"rev_{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "original_task_id": task_id,
            "original_task_title": task_title,
            "subject_tag": subject_tag or "",
            "key_concepts": key_concepts or [],
            "review_number": 1,
            "ease_factor": initial_ease,
            "interval_days": first_interval,
            "next_review_date": next_review,
            "last_review_date": None,
            "quality_history": [],
            "status": "scheduled",
            "created_at": now,
        }

        try:
            self._db[REVIEW_COLLECTION].insert_one(review_doc)
            logger.info(
                "review_scheduled",
                extra={
                    "user_id": user_id,
                    "task_id": task_id,
                    "review_number": 1,
                    "next_review": next_review.isoformat(),
                    "interval_days": first_interval,
                },
            )
            return review_doc
        except Exception as exc:
            logger.error("review_schedule_failed", extra={"error": str(exc)})
            return None

    def record_review_result(
        self,
        user_id: str,
        review_id: str,
        quality_score: int,
    ) -> Optional[Dict]:
        """
        Record the result of a review and schedule the next one.

        Args:
            user_id:       User identifier.
            review_id:     The review document ID.
            quality_score: User self-reported recall quality (0-5).
                          0 = complete blackout
                          1 = incorrect, but recognized correct answer
                          2 = incorrect, correct answer seemed easy to recall
                          3 = correct with significant difficulty
                          4 = correct after hesitation
                          5 = perfect, instant recall

        Returns:
            Dict with updated review info and next schedule, or None.
        """
        if self._db is None:
            return None

        quality_score = max(0, min(5, quality_score))

        try:
            review = self._db[REVIEW_COLLECTION].find_one(
                {
                    "review_id": review_id,
                    "user_id": user_id,
                }
            )

            if not review:
                logger.warning("review_not_found", extra={"review_id": review_id})
                return None

            now = datetime.now(timezone.utc)
            old_ease = review["ease_factor"]
            old_review_number = review["review_number"]

            # Calculate new ease factor based on quality
            new_ease = self._calculate_ease_factor(old_ease, quality_score)

            # Calculate next interval
            if quality_score < 2:
                # Failed recall: reset to beginning
                new_review_number = 1
                new_interval = BASE_INTERVALS[0]
                logger.info(
                    "review_reset",
                    extra={
                        "review_id": review_id,
                        "quality": quality_score,
                    },
                )
            else:
                new_review_number = old_review_number + 1
                new_interval = self._calculate_interval(new_review_number, new_ease)

            next_review = now + timedelta(days=new_interval)

            # Update quality history
            quality_history = review.get("quality_history", [])
            quality_history.append(quality_score)

            # Mark current review as completed and schedule next
            self._db[REVIEW_COLLECTION].update_one(
                {"review_id": review_id},
                {
                    "$set": {
                        "status": "completed",
                        "last_review_date": now,
                        "quality_history": quality_history,
                    }
                },
            )

            # Create next review document
            next_review_doc = {
                "review_id": f"rev_{uuid.uuid4().hex[:12]}",
                "user_id": user_id,
                "original_task_id": review["original_task_id"],
                "original_task_title": review["original_task_title"],
                "subject_tag": review["subject_tag"],
                "key_concepts": review["key_concepts"],
                "review_number": new_review_number,
                "ease_factor": new_ease,
                "interval_days": new_interval,
                "next_review_date": next_review,
                "last_review_date": now,
                "quality_history": quality_history,
                "status": "scheduled",
                "created_at": now,
            }

            self._db[REVIEW_COLLECTION].insert_one(next_review_doc)

            logger.info(
                "review_completed_next_scheduled",
                extra={
                    "user_id": user_id,
                    "review_id": review_id,
                    "quality": quality_score,
                    "old_ease": old_ease,
                    "new_ease": new_ease,
                    "next_interval": new_interval,
                    "next_review": next_review.isoformat(),
                },
            )

            return {
                "completed_review_id": review_id,
                "quality_score": quality_score,
                "ease_factor": new_ease,
                "next_review_id": next_review_doc["review_id"],
                "next_review_date": next_review.isoformat(),
                "next_interval_days": new_interval,
                "review_number": new_review_number,
            }

        except Exception as exc:
            logger.error("review_result_failed", extra={"error": str(exc)})
            return None

    def get_pending_reviews(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[Dict]:
        """
        Get all pending/overdue reviews for a user, ordered by urgency.

        Reviews that are past their next_review_date are returned first.

        Args:
            user_id: User identifier.
            limit:   Max number of reviews to return.

        Returns:
            List of review documents sorted by next_review_date (oldest first).
        """
        if self._db is None:
            return []

        try:
            now = datetime.now(timezone.utc)

            # Mark overdue reviews
            self._db[REVIEW_COLLECTION].update_many(
                {
                    "user_id": user_id,
                    "status": "scheduled",
                    "next_review_date": {"$lt": now},
                },
                {"$set": {"status": "overdue"}},
            )

            # Fetch pending and overdue reviews
            reviews = list(
                self._db[REVIEW_COLLECTION]
                .find(
                    {
                        "user_id": user_id,
                        "status": {"$in": ["scheduled", "overdue"]},
                    }
                )
                .sort("next_review_date", 1)
                .limit(limit)
            )

            # Convert ObjectId to string for JSON serialization
            for review in reviews:
                review["_id"] = str(review["_id"])
                if review.get("next_review_date"):
                    review["next_review_date"] = review["next_review_date"].isoformat()
                if review.get("last_review_date"):
                    review["last_review_date"] = review["last_review_date"].isoformat()
                if review.get("created_at"):
                    review["created_at"] = review["created_at"].isoformat()

            return reviews

        except Exception as exc:
            logger.error("get_pending_reviews_failed", extra={"error": str(exc)})
            return []

    def get_review_stats(self, user_id: str) -> Dict:
        """
        Get review statistics for a user.

        Returns:
            Dict with review counts and performance metrics.
        """
        if self._db is None:
            return {
                "total": 0,
                "pending": 0,
                "overdue": 0,
                "completed": 0,
                "avg_quality": 0,
            }

        try:
            collection = self._db[REVIEW_COLLECTION]
            now = datetime.now(timezone.utc)

            total = collection.count_documents({"user_id": user_id})
            pending = collection.count_documents(
                {
                    "user_id": user_id,
                    "status": "scheduled",
                    "next_review_date": {"$gte": now},
                }
            )
            overdue = collection.count_documents(
                {
                    "user_id": user_id,
                    "status": {"$in": ["scheduled", "overdue"]},
                    "next_review_date": {"$lt": now},
                }
            )
            completed = collection.count_documents(
                {
                    "user_id": user_id,
                    "status": "completed",
                }
            )

            # Calculate average quality score
            completed_reviews = list(
                collection.find(
                    {
                        "user_id": user_id,
                        "status": "completed",
                        "quality_history": {"$ne": []},
                    }
                ).limit(100)
            )

            all_qualities = []
            for review in completed_reviews:
                all_qualities.extend(review.get("quality_history", []))

            avg_quality = (
                round(sum(all_qualities) / len(all_qualities), 1)
                if all_qualities
                else 0
            )

            return {
                "total": total,
                "pending": pending,
                "overdue": overdue,
                "completed": completed,
                "avg_quality": avg_quality,
            }

        except Exception as exc:
            logger.error("review_stats_failed", extra={"error": str(exc)})
            return {
                "total": 0,
                "pending": 0,
                "overdue": 0,
                "completed": 0,
                "avg_quality": 0,
            }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _calculate_ease_factor(current_ease: float, quality: int) -> float:
        """
        Calculate new ease factor based on SM-2 algorithm.

        Args:
            current_ease: Current ease factor.
            quality:      Recall quality score (0-5).

        Returns:
            New ease factor clamped to [MIN_EASE, MAX_EASE].
        """
        if quality < 2:
            new_ease = current_ease - 0.3
        elif quality == 2:
            new_ease = current_ease - 0.15
        elif quality == 3:
            new_ease = current_ease
        elif quality == 4:
            new_ease = current_ease + 0.1
        else:  # quality == 5
            new_ease = current_ease + 0.15

        return max(MIN_EASE, min(MAX_EASE, round(new_ease, 2)))

    @staticmethod
    def _calculate_interval(review_number: int, ease_factor: float) -> int:
        """
        Calculate next review interval in days.

        Uses base Fibonacci-like intervals for first 6 reviews,
        then multiplies by ease factor for subsequent reviews.

        Args:
            review_number: Which review iteration (1-based).
            ease_factor:   Current ease factor.

        Returns:
            Interval in days (minimum 1, maximum 180).
        """
        if review_number <= len(BASE_INTERVALS):
            interval = BASE_INTERVALS[review_number - 1]
        else:
            # Beyond base intervals: last base * ease_factor^(review_number - len(BASE_INTERVALS))
            extra_steps = review_number - len(BASE_INTERVALS)
            interval = int(BASE_INTERVALS[-1] * (ease_factor**extra_steps))

        # Clamp to reasonable bounds
        return max(1, min(interval, 180))
