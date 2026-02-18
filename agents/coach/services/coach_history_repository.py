"""
CoachHistoryRepository — persist and retrieve coach actions from MongoDB.

Collection: `coach_actions`

Document schema:
  {
    user_id:     str,
    trace_id:    str,
    ts:          datetime (UTC),
    action_type: str,   # nudge | encourage | suggest_break | ...
    message:     str | null,
    reasoning:   str,
    focus_state: str,
    fatigue_state: str,
    affective_state: str,
  }
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

COACH_ACTIONS_COLLECTION = "coach_actions"
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

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def save_action(
        self,
        user_id: str,
        action,  # CoachAction
        coach_input,  # CoachInput
        trace_id: str = "",
    ) -> None:
        """
        Persist a CoachAction alongside key signal context.

        Args:
            user_id:     User the action was generated for.
            action:      CoachAction Pydantic model.
            coach_input: The CoachInput that triggered the action (for context).
            trace_id:    Optional request trace ID for log correlation.
        """
        if self._db is None:
            return
        try:
            doc: Dict = {
                "user_id": user_id,
                "trace_id": trace_id,
                "ts": datetime.now(tz=timezone.utc),
                "action_type": action.action_type,
                "message": action.message,
                "reasoning": action.reasoning,
                "focus_state": coach_input.focus_state.state,
                "fatigue_state": coach_input.fatigue_state.state,
                "affective_state": coach_input.affective_state,
            }
            self._db[COACH_ACTIONS_COLLECTION].insert_one(doc)
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
