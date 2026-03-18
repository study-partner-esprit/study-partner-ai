"""
PacingStore — personalised pacing factor derived from historical task completions.

Stores individual task completion ratios (actual / estimated) in MongoDB
collection `pacing_data` and returns a rolling median as the pacing factor
used by the Planner to scale task durations.

Schema (one document per completion record):
  {
    user_id:     str,
    task_id:     str,
    subject_tag: str,   # e.g. "math", "programming", "" for generic
    estimated:   int,   # minutes
    actual:      int,   # minutes
    ratio:       float, # actual / estimated
    ts:          datetime
  }
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

PACING_COLLECTION = "pacing_data"
MAX_RECORDS = 20  # rolling window per (user, subject)
DEFAULT_FACTOR = 1.0
MIN_RECORDS = 3  # minimum records needed before we trust the estimate


class PacingStore:
    """
    Stores and retrieves pacing factors from MongoDB.

    Falls back gracefully to DEFAULT_FACTOR (1.0) when MongoDB is unavailable
    or there is insufficient history.
    """

    def __init__(self) -> None:
        self._db = None
        try:
            from services.database import get_db

            self._db = get_db()
        except Exception as exc:
            logger.warning("pacing_store_no_db", extra={"error": str(exc)})

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_user_pacing_factor(
        self,
        user_id: str,
        subject_tag: str = "",
    ) -> float:
        """
        Return the median actual/estimated ratio for *user_id*.

        Args:
            user_id:     Unique user identifier.
            subject_tag: Optional subject filter (e.g. "math").
                         Falls back to global history if no subject-specific
                         records exist.

        Returns:
            Pacing factor ≥ 0.5.  Values > 1.0 mean the user is slower
            than estimated; < 1.0 means faster.
        """
        if not user_id or self._db is None:
            return DEFAULT_FACTOR

        factor = self._compute_factor(user_id, subject_tag)
        if factor is None and subject_tag:
            # Fall back to global (no subject filter)
            factor = self._compute_factor(user_id, "")
        if factor is None:
            return DEFAULT_FACTOR
        # Clamp to a sensible range to avoid absurd durations
        return max(0.5, min(factor, 3.0))

    def record_task_completion(
        self,
        user_id: str,
        task_id: str,
        estimated_minutes: int,
        actual_minutes: int,
        subject_tag: str = "",
    ) -> None:
        """
        Persist a completion record and prune old records beyond MAX_RECORDS.

        Args:
            user_id:           Unique user identifier.
            task_id:           Task that was completed.
            estimated_minutes: How long the planner thought it would take.
            actual_minutes:    How long it actually took.
            subject_tag:       Subject area (used for subject-specific pacing).
        """
        if not user_id or self._db is None:
            return
        if estimated_minutes <= 0 or actual_minutes <= 0:
            return

        ratio = actual_minutes / estimated_minutes
        doc = {
            "user_id": user_id,
            "task_id": task_id,
            "subject_tag": subject_tag,
            "estimated": estimated_minutes,
            "actual": actual_minutes,
            "ratio": ratio,
            "ts": datetime.now(tz=timezone.utc),
        }
        try:
            col = self._db[PACING_COLLECTION]
            col.insert_one(doc)
            # Prune: keep only the last MAX_RECORDS per (user, subject)
            pipeline = [
                {"$match": {"user_id": user_id, "subject_tag": subject_tag}},
                {"$sort": {"ts": -1}},
                {"$skip": MAX_RECORDS},
            ]
            old_docs = list(col.aggregate(pipeline))
            if old_docs:
                old_ids = [d["_id"] for d in old_docs]
                col.delete_many({"_id": {"$in": old_ids}})
            logger.info(
                "pacing_store_recorded",
                extra={
                    "user_id": user_id,
                    "ratio": round(ratio, 3),
                    "subject_tag": subject_tag,
                },
            )
        except Exception as exc:
            logger.warning("pacing_store_record_error", extra={"error": str(exc)})

    def update_from_execution(
        self,
        user_id: str,
        planned_minutes: int,
        actual_minutes: int,
        task_id: str = "",
        subject_tag: str = "",
    ) -> None:
        """
        Compatibility shim — delegates to record_task_completion.
        Called by PlannerAgent after building a plan.
        """
        self.record_task_completion(
            user_id, task_id, planned_minutes, actual_minutes, subject_tag
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_factor(self, user_id: str, subject_tag: str) -> Optional[float]:
        """
        Query MongoDB and compute the median ratio.  Returns None if
        fewer than MIN_RECORDS documents exist for this (user, subject).
        """
        try:
            col = self._db[PACING_COLLECTION]
            query: dict = {"user_id": user_id}
            if subject_tag:
                query["subject_tag"] = subject_tag
            records = list(
                col.find(query, {"ratio": 1}).sort("ts", -1).limit(MAX_RECORDS)
            )
            if len(records) < MIN_RECORDS:
                return None
            ratios = [r["ratio"] for r in records]
            return statistics.median(ratios)
        except Exception as exc:
            logger.warning(
                "pacing_store_compute_error",
                extra={"user_id": user_id, "error": str(exc)},
            )
            return None
