"""
Unit tests for CoachHistoryRepository.

Run with:
    pytest agents/coach/tests/test_coach_history.py -v
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents.coach.models.schemas import (
    CoachAction,
    CoachInput,
    FocusState,
    FatigueState,
)
from agents.coach.services.coach_history_repository import (
    COACH_ACTIONS_TRACE_INDEX,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeCol:
    """Stateful stand-in for a pymongo Collection (keyed on the filter)."""

    def __init__(self):
        self.docs = {}

    def update_one(self, flt, update, upsert=True):
        key = flt["trace_id"]
        if key in self.docs:
            return SimpleNamespace(matched_count=1, upserted_id=None)
        if upsert:
            self.docs[key] = dict(update["$setOnInsert"])
            return SimpleNamespace(matched_count=0, upserted_id="obj-1")
        raise AssertionError("upsert must be enabled")


def _make_coach_input(**overrides):
    defaults = dict(
        scheduled_tasks=[],
        current_time=datetime.now(tz=timezone.utc),
        focus_state=FocusState(state="Drifting", score=0.5),
        fatigue_state=FatigueState(state="Moderate", score=0.3),
        affective_state="engaged",
    )
    defaults.update(overrides)
    return CoachInput(**defaults)


def _make_action(**overrides):
    defaults = dict(
        action_type="nudge",
        message="Focus up!",
        reasoning="Drifting.",
    )
    defaults.update(overrides)
    return CoachAction(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCoachHistoryRepository:

    def _make_repo(self, collection):
        """Return a repo instance with its MongoDB collection mocked (bypasses __init__)."""
        from agents.coach.services.coach_history_repository import (
            CoachHistoryRepository,
        )

        repo = CoachHistoryRepository.__new__(CoachHistoryRepository)
        repo._db = {"coach_actions": collection}
        return repo

    def test_save_action_inserts_document(self):
        col = MagicMock()
        col.update_one.return_value = SimpleNamespace(
            matched_count=0, upserted_id="obj-1"
        )
        repo = self._make_repo(col)

        action = _make_action()
        coach_input = _make_coach_input()

        repo.save_action(
            "user_1", action, coach_input, trace_id="t-123", correlation_id="c-123"
        )

        col.update_one.assert_called_once()
        flt, update, kwargs = (
            col.update_one.call_args[0][0],
            col.update_one.call_args[0][1],
            col.update_one.call_args[1],
        )
        assert flt == {"trace_id": "t-123"}
        assert kwargs.get("upsert") is True
        doc = update["$setOnInsert"]
        assert doc["user_id"] == "user_1"
        assert doc["action_type"] == "nudge"
        assert doc["trace_id"] == "t-123"
        assert doc["correlation_id"] == "c-123"

    def test_save_action_idempotent_by_trace_id(self):
        col = _FakeCol()
        repo = self._make_repo(col)
        coach_input = _make_coach_input()

        repo.save_action("user_1", _make_action(), coach_input, trace_id="job-1")
        # Same correlation id redelivered (retry succeeded after an earlier
        # attempt) → no duplicate row; the first write wins.
        repo.save_action(
            "user_1",
            _make_action(action_type="encourage"),
            coach_input,
            trace_id="job-1",
            correlation_id="job-1",
        )

        assert len(col.docs) == 1
        assert col.docs["job-1"]["action_type"] == "nudge"
        assert col.docs["job-1"]["correlation_id"] == "job-1"

        repo.save_action("user_1", _make_action(), coach_input, trace_id="job-2")
        assert len(col.docs) == 2

    def test_ensures_unique_trace_id_index(self):
        col = MagicMock()
        repo = self._make_repo(col)

        repo._ensure_index()

        col.create_index.assert_called_once()
        args, kwargs = col.create_index.call_args
        assert args[0] == [("trace_id", 1)]
        assert kwargs.get("unique") is True
        assert kwargs.get("name") == COACH_ACTIONS_TRACE_INDEX

    def test_save_action_no_db_silently_skips(self):
        from agents.coach.services.coach_history_repository import (
            CoachHistoryRepository,
        )

        repo = CoachHistoryRepository.__new__(CoachHistoryRepository)
        repo._db = None

        # Should not raise
        repo.save_action("u", _make_action(), _make_coach_input())

    def test_get_recent_actions_returns_list(self):
        col = MagicMock()
        expected = [
            {
                "ts": datetime.now(tz=timezone.utc),
                "action_type": "encourage",
                "message": "Great!",
                "focus_state": "Focused",
                "fatigue_state": "Alert",
                "affective_state": "confident",
            },
        ]
        col.find.return_value.sort.return_value.limit.return_value = expected

        repo = self._make_repo(col)
        result = repo.get_recent_actions("user_1", limit=5)

        assert result == expected
        col.find.assert_called_once_with(
            {"user_id": "user_1"},
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

    def test_get_recent_actions_empty_user_returns_empty(self):
        from agents.coach.services.coach_history_repository import (
            CoachHistoryRepository,
        )

        repo = CoachHistoryRepository.__new__(CoachHistoryRepository)
        repo._db = MagicMock()

        result = repo.get_recent_actions("")
        assert result == []

    def test_get_recent_actions_no_db_returns_empty(self):
        from agents.coach.services.coach_history_repository import (
            CoachHistoryRepository,
        )

        repo = CoachHistoryRepository.__new__(CoachHistoryRepository)
        repo._db = None

        result = repo.get_recent_actions("user_1")
        assert result == []
