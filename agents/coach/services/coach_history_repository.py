"""
CoachHistoryRepository — persist and retrieve coach actions from MongoDB.

Collection: `coach_actions`

Document schema:
  {
    user_id:        str,
    trace_id:       str,          # COACH-09 idempotency key (= job correlationId)
    correlation_id: str,          # explicit correlation id (same value as trace_id
                                  # at the worker boundary)
    ts:             datetime (UTC),
    action_type:    str,   # nudge | encourage | suggest_break | ...
    message:        str | null,
    reasoning:      str,
    focus_state:    str,
    fatigue_state:  str,
    affective_state: str,
  }

Persistence is idempotent by `trace_id`: a completed job is persisted exactly
once, so retried-then-succeeded or redelivered jobs cannot duplicate history
rows (COACH-09 AC#2).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

COACH_ACTIONS_COLLECTION = "coach_actions"
COACH_ACTIONS_TRACE_INDEX = "uniq_coach_actions_trace_id"
DEFAULT_HISTORY_LIMIT = 5


class CoachHistoryRepository:
    """
    Thin wrapper around the `coach_actions` MongoDB collection.

    Falls back gracefully to no-ops when MongoDB is unavailable.
    """

    def __init__(self) -> None:
        self._db = None
        try:
            from services.database import get_db

            self._db = get_db()
        except Exception as exc:
            logger.warning("coach_history_no_db", extra={"error": str(exc)})
        self._ensure_index()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def _ensure_index(self) -> None:
        """Best-effort unique index on `trace_id` — the COACH-09 key.

        The upsert in `save_action` is logically idempotent; the unique index
        is the database-level guarantee against a race writing a duplicate
        history row for the same correlation id.
        """
        if self._db is None:
            return
        try:
            self._db[COACH_ACTIONS_COLLECTION].create_index(
                [("trace_id", 1)],
                unique=True,
                name=COACH_ACTIONS_TRACE_INDEX,
            )
        except Exception as exc:
            logger.warning(
                "coach_history_index_error",
                extra={"error": str(exc)},
            )

    def save_action(
        self,
        user_id: str,
        action,  # CoachAction
        coach_input,  # CoachInput
        trace_id: str = "",
        correlation_id: str = "",
    ) -> None:
        """
        Persist a CoachAction alongside key signal context.

        Idempotent by `trace_id` (the job correlation id at the worker
        boundary): re-persisting a completed job is a no-op, so a retried or
        redelivered job cannot create a duplicate history row (COACH-09).

        Args:
            user_id:        User the action was generated for.
            action:         CoachAction Pydantic model.
            coach_input:    The CoachInput that triggered the action (for context).
            trace_id:       Correlation key; the worker passes `envelope.correlationId`.
            correlation_id: Optional explicit correlation id; defaults to
                            `trace_id` when empty.
        """
        if self._db is None:
            return
        try:
            correlation_id = correlation_id or trace_id
            doc: Dict = {
                "user_id": user_id,
                "trace_id": trace_id,
                "correlation_id": correlation_id,
                "ts": datetime.now(tz=timezone.utc),
                "action_type": action.action_type,
                "message": action.message,
                "reasoning": action.reasoning,
                "focus_state": coach_input.focus_state.state,
                "fatigue_state": coach_input.fatigue_state.state,
                "affective_state": coach_input.affective_state,
            }
            col = self._db[COACH_ACTIONS_COLLECTION]
            # Insert only when no record with the same trace_id exists yet;
            # a match is a no-op (the original write is preserved).
            col.update_one(
                {"trace_id": trace_id},
                {"$setOnInsert": doc},
                upsert=True,
            )
            logger.info(
                "coach_action_saved",
                extra={
                    "user_id": user_id,
                    "action_type": action.action_type,
                    "trace_id": trace_id,
                },
            )
        except Exception as exc:
            logger.warning("coach_action_save_error", extra={"error": str(exc)})

    def get_recent_actions(
        self,
        user_id: str,
        limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> List[Dict]:
        """
        Return the *limit* most recent coach actions for *user_id*.

        Args:
            user_id: User to query.
            limit:   Number of records to return (default 5).

        Returns:
            List of dicts (newest first), each with keys:
            ts, action_type, message, focus_state, fatigue_state, affective_state.
        """
        if self._db is None or not user_id:
            return []
        try:
            col = self._db[COACH_ACTIONS_COLLECTION]
            docs = list(
                col.find(
                    {"user_id": user_id},
                    {
                        "_id": 0,
                        "ts": 1,
                        "action_type": 1,
                        "message": 1,
                        "focus_state": 1,
                        "fatigue_state": 1,
                        "affective_state": 1,
                    },
                )
                .sort("ts", -1)
                .limit(limit)
            )
            return docs
        except Exception as exc:
            logger.warning(
                "coach_history_read_error",
                extra={"user_id": user_id, "error": str(exc)},
            )
            return []
