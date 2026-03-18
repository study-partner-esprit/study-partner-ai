"""
Unit tests for PacingStore.

Run with:
    pytest agents/planner/tests/test_pacing_store.py -v
"""

from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

import pytest

from agents.planner.memory.pacing_store import PacingStore, MIN_RECORDS, DEFAULT_FACTOR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(documents=None):
    """Return a PacingStore with a mocked MongoDB collection."""
    col = MagicMock()
    if documents is not None:
        col.find.return_value.sort.return_value.limit.return_value = documents
    store = PacingStore.__new__(PacingStore)
    store._db = {"pacing_data": col}
    return store, col


# ---------------------------------------------------------------------------
# Tests: get_user_pacing_factor
# ---------------------------------------------------------------------------


class TestGetUserPacingFactor:

    def test_returns_default_when_no_user_id(self):
        store, _ = _make_store()
        assert store.get_user_pacing_factor("") == DEFAULT_FACTOR

    def test_returns_default_when_no_db(self):
        store = PacingStore.__new__(PacingStore)
        store._db = None
        assert store.get_user_pacing_factor("u1") == DEFAULT_FACTOR

    def test_returns_default_with_insufficient_records(self):
        # Fewer than MIN_RECORDS
        docs = [{"ratio": 1.2}] * (MIN_RECORDS - 1)
        store, _ = _make_store(docs)
        assert store.get_user_pacing_factor("u1") == DEFAULT_FACTOR

    def test_computes_median_with_enough_records(self):
        docs = [{"ratio": r} for r in [1.5, 1.0, 2.0, 1.5, 1.0]]  # median = 1.5
        store, _ = _make_store(docs)
        factor = store.get_user_pacing_factor("u1")
        assert factor == pytest.approx(1.5)

    def test_clamps_to_maximum(self):
        docs = [{"ratio": 10.0}] * (MIN_RECORDS + 1)
        store, _ = _make_store(docs)
        assert store.get_user_pacing_factor("u1") == 3.0  # capped

    def test_clamps_to_minimum(self):
        docs = [{"ratio": 0.0}] * (MIN_RECORDS + 1)
        store, _ = _make_store(docs)
        assert store.get_user_pacing_factor("u1") == 0.5  # capped

    def test_falls_back_to_global_when_subject_empty(self):
        global_docs = [{"ratio": 1.2}] * (MIN_RECORDS + 1)
        store, col = _make_store()

        call_results = [
            [],  # subject-specific: empty
            global_docs,  # global fallback
        ]
        call_count = [0]

        def fake_find(query, projection):
            result = MagicMock()
            result.sort.return_value.limit.return_value = call_results[call_count[0]]
            call_count[0] += 1
            return result

        col.find = fake_find
        factor = store.get_user_pacing_factor("u1", "math")
        assert factor == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# Tests: record_task_completion
# ---------------------------------------------------------------------------


class TestRecordTaskCompletion:

    def test_inserts_document(self):
        store, col = _make_store([])
        col.find.return_value.sort.return_value.limit.return_value = []

        store.record_task_completion("u1", "t1", 30, 45, "math")

        col.insert_one.assert_called_once()
        doc = col.insert_one.call_args[0][0]
        assert doc["user_id"] == "u1"
        assert doc["estimated"] == 30
        assert doc["actual"] == 45
        assert abs(doc["ratio"] - 1.5) < 0.001
        assert doc["subject_tag"] == "math"

    def test_skips_when_no_db(self):
        store = PacingStore.__new__(PacingStore)
        store._db = None
        # Should not raise
        store.record_task_completion("u1", "t1", 30, 45)

    def test_skips_invalid_times(self):
        store, col = _make_store()
        store.record_task_completion("u1", "t1", 0, 45)
        col.insert_one.assert_not_called()

    def test_prunes_old_records(self):
        old_docs = [{"_id": f"id_{i}"} for i in range(5)]
        col = MagicMock()
        col.find.return_value.sort.return_value.limit.return_value = (
            []
        )  # for find in record
        col.aggregate.return_value = old_docs

        store = PacingStore.__new__(PacingStore)
        store._db = {"pacing_data": col}

        store.record_task_completion("u1", "t1", 30, 45)

        col.delete_many.assert_called_once()
        delete_ids = col.delete_many.call_args[0][0]["_id"]["$in"]
        assert delete_ids == ["id_0", "id_1", "id_2", "id_3", "id_4"]
