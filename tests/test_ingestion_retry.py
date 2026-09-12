"""INGEST-08 — ingestion retry / DLQ policy (no broker, no Mongo, no FAISS).

The retry backoff / DLQ machinery lives in ``BaseAIWorker`` (AI-COM-06); these
tests pin the four INGEST-08 acceptance criteria AT the ingestion boundary:

1. OCR/LLM failures are RETRYABLE — scheduled with backoff, never DLQ'd early,
   and only escalate to the DLQ once retries are exhausted.
2. Schema/file validation failures are TERMINAL — straight to the DLQ at
   attempt 0, no retry is ever scheduled.
3. The DLQ replay path re-runs an ingest job with a fresh messageId and gets a
   full retry budget again (validated on the Node side by dlq-replay.test.js).
4. ``VectorStoreAdapter.add_course`` is idempotent on replay — Mongo upserts by
   course_id with ``$set`` (replace, never append) and FAISS rebuilds per
   course, so a replayed job never duplicates embeddings.
"""

from __future__ import annotations

import sys
import types
import uuid
from typing import Dict, List, Optional

import numpy as np
import pytest

from messaging.failures import FailureClass, TerminalError, classify_failure
from tests.test_base_worker import FakeMessage
from workers.idempotency import InMemoryIdempotencyStore
from workers.ingestion_worker import IngestionWorker

UUID_B = "0f8e2d1a-3b4c-4d6e-8f80-91a2b3c4d5e6"

MAX_RETRIES = 3  # RETRY_DELAYS_MS default ladder [1000, 4000, 16000]

OCR_RETRYABLE = "tesseract OCR daemon timed out"
LLM_RETRYABLE = "openai embedding service rate limit exceeded"


def ingest_envelope(**overrides) -> dict:
    base = {
        "messageId": str(uuid.uuid4()),
        "correlationId": UUID_B,
        "type": "study.ingest.course",
        "version": "1",
        "userId": "user-1",
        "requestId": "req-ingest",
        "timestamp": "2026-08-19T08:00:00Z",
        "payload": {
            "courseId": "course-1",
            "fileRef": "uploads/courses/course-1",
            "files": [
                {
                    "filename": "lecture-notes-c8f2.pdf",
                    "originalName": "lecture-notes.pdf",
                    "mimetype": "application/pdf",
                    "size": 2048,
                    "path": "courses/course-1/lecture-notes-c8f2.pdf",
                }
            ],
        },
    }
    base.update(overrides)
    return base


@pytest.fixture
def worker() -> IngestionWorker:
    w = IngestionWorker(
        idempotency_store=InMemoryIdempotencyStore(),
        uploads_root="/tmp/opencode/ingest-retry",
    )
    w.recorded_results: List = []
    w.recorded_progress: List = []

    async def _result(env, *, status, payload=None, error=None):
        w.recorded_results.append((status, payload, error))

    async def _progress(env, *, stage, progress=None, detail=None):
        w.recorded_progress.append((stage, progress, detail))

    w._publish_result = _result
    w._publish_progress = _progress
    return w


async def _capture_republish(worker):
    """Swap in a recorder for _republish_for_retry (mirrors test_base_worker)."""
    scheduled = {}

    async def fake_republish(message, env, next_attempt, delay_ms, reason):
        scheduled["next"] = next_attempt
        scheduled["delay"] = delay_ms
        scheduled["reason"] = reason
        message.ack()

    worker._republish_for_retry = fake_republish
    return scheduled


# ------------------------------------------------ AC 1: OCR/LLM are retryable


def test_ocr_and_llm_failures_classify_retryable():
    for msg in (OCR_RETRYABLE, LLM_RETRYABLE, "connection closed", "HTTP 503"):
        assert classify_failure(RuntimeError(msg)) is FailureClass.RETRYABLE, msg
    assert (
        classify_failure(TimeoutError("pdf parse timed out")) is FailureClass.RETRYABLE
    )


async def test_ocr_failure_schedules_backoff_retry_not_dlq(worker, monkeypatch):
    def ocr_fail(self, request):
        raise RuntimeError(OCR_RETRYABLE)  # surfaced by the parsing stage

    monkeypatch.setattr(worker, "_parse_stage", ocr_fail.__get__(worker))
    scheduled = await _capture_republish(worker)

    msg = FakeMessage(ingest_envelope(), headers={"x-retry-count": 0})
    await worker.on_message(msg)

    assert scheduled == {"next": 1, "delay": 1000, "reason": OCR_RETRYABLE}
    assert msg.acked and not msg.nacked  # original ACKed; delayed copy continues
    assert worker.recorded_results == []  # no failed result before exhaustion
    assert worker.recorded_progress[0][0] == "parsing"  # stage recorded first


async def test_llm_enrichment_failure_schedules_backoff_retry(worker, monkeypatch):
    def parse_stub(self, request):
        return [
            {"title": "t", "full_content": "content", "key_concepts": []}, 1, ["n.pdf"]
        ]

    def enrich_fail(self, subtopics):
        raise TimeoutError(LLM_RETRYABLE)

    monkeypatch.setattr(worker, "_parse_stage", parse_stub.__get__(worker))
    monkeypatch.setattr(worker, "_enrich_stage", enrich_fail.__get__(worker))
    scheduled = await _capture_republish(worker)

    msg = FakeMessage(ingest_envelope(), headers={"x-retry-count": 0})
    await worker.on_message(msg)

    assert scheduled == {"next": 1, "delay": 1000, "reason": LLM_RETRYABLE}
    assert msg.acked and not msg.nacked
    assert worker.recorded_results == []
    assert worker.recorded_progress[1][0] == "enriching"  # got past parsing


async def test_retry_exhaustion_dlqs_with_failed_result(worker, monkeypatch):
    def ocr_fail(self, request):
        raise RuntimeError("OCR still timing out")

    monkeypatch.setattr(worker, "_parse_stage", ocr_fail.__get__(worker))

    msg = FakeMessage(ingest_envelope(), headers={"x-retry-count": MAX_RETRIES})
    with pytest.raises(TerminalError):
        await worker.on_message(msg)

    assert msg.nacked and msg.requeue is False  # DLX → ai.dlq.study.ingest.course
    assert worker.recorded_results[0][0] == "failed"
    assert "timing out" in worker.recorded_results[0][2]


# ------------------------------ AC 2: schema/file validation → DLQ immediately


async def test_invalid_schema_dlqs_immediately_without_retry(worker):
    scheduled = await _capture_republish(worker)

    bad = ingest_envelope(
        payload={"courseId": "course-1"}  # missing fileRef/files → schema error
    )
    msg = FakeMessage(bad, headers={"x-retry-count": 0})
    with pytest.raises(TerminalError):
        await worker.on_message(msg)

    assert scheduled == {}  # no retry scheduled at attempt 0
    assert msg.nacked and msg.requeue is False  # straight to the DLQ
    assert worker.recorded_results[0][0] == "failed"
    assert "invalid ingest payload" in worker.recorded_results[0][2]
    assert worker.recorded_progress == []  # validation happens before any stage


async def test_file_validation_dlqs_immediately_without_retry(worker, monkeypatch):
    scheduled = await _capture_republish(worker)

    def utf8_fail(self, request):
        fname = request.files[0].filename
        raise TerminalError(f"text file is not valid UTF-8: {fname}")

    monkeypatch.setattr(worker, "_parse_stage", utf8_fail.__get__(worker))

    msg = FakeMessage(ingest_envelope(), headers={"x-retry-count": 0})
    with pytest.raises(TerminalError):
        await worker.on_message(msg)

    assert scheduled == {}
    assert msg.nacked and msg.requeue is False
    assert worker.recorded_results[0][0] == "failed"


# ----------------------------------------- AC 3: replay re-runs with full budget


async def test_replayed_dlq_message_gets_fresh_attempt_budget(worker, monkeypatch):
    """A DLQ'd message re-published (fresh messageId per dlq-replay.js) must be
    processed as attempt 0 — not swallowed by the original claim."""
    def ocr_fail(self, request):
        raise RuntimeError(OCR_RETRYABLE)

    monkeypatch.setattr(worker, "_parse_stage", ocr_fail.__get__(worker))
    scheduled = await _capture_republish(worker)

    replayed = FakeMessage(ingest_envelope(), headers={"x-retry-count": 0})
    await worker.on_message(replayed)
    assert scheduled == {"next": 1, "delay": 1000, "reason": OCR_RETRYABLE}


# ------------------------------------- AC 4: no duplicate embeddings on replay


class _FakeIndex:
    DIM = 384

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.ntotal = 0

    def add(self, rows) -> None:  # fresh rebuild per add_course call
        self.ntotal = len(rows)


class _FakeFaiss:
    IndexFlatIP = _FakeIndex

    @staticmethod
    def write_index(index, path) -> None:  # pragma: no cover - no-op fake
        return None


class _FakeColl:
    def __init__(self):
        self.docs: Dict[str, dict] = {}

    def find_one(self, query: dict) -> Optional[dict]:
        return self.docs.get(query.get("course_id"))

    def update_one(self, query: dict, update: dict, upsert: bool = False) -> None:
        cid = query["course_id"]
        merged = {**self.docs.get(cid, {}), **update.get("$set", {})}
        self.docs[cid] = merged


class _FakeDB:
    _COLL = "chunk_embeddings"

    def __init__(self):
        self.colls = {self._COLL: _FakeColl()}

    def __getitem__(self, key):
        return self.colls[key]


def test_add_course_rebuilds_idempotently_on_replay(tmp_path, monkeypatch):
    """Simulate original ingest job then a DLQ replay of the same course:
    the second add_course (fresh messageId, same course) REPLACES the Mongo
    document and rebuilds the FAISS index — no duplicate embeddings."""
    monkeypatch.setitem(sys.modules, "faiss", _FakeFaiss())

    import services.vector_store.adapter as adapter

    monkeypatch.setattr(adapter, "FAISS_INDEX_DIR", tmp_path)

    db = _FakeDB()
    vs = adapter.VectorStoreAdapter(db)

    first = np.ones((2, 384), dtype="float32")
    vs.add_course("course-1", ["c1", "c2"], first, [{"course_id": "course-1"}] * 2)
    assert vs._indices["course-1"]["index"].ntotal == 2

    # The replayed job ends up with the same chunks → identical vectors.
    replay = np.asarray(first, dtype="float32")
    vs.add_course("course-1", ["c1", "c2"], replay, [{"course_id": "course-1"}] * 2)

    assert vs._indices["course-1"]["index"].ntotal == 2  # rebuilt, not accumulated
    assert len(db.colls["chunk_embeddings"].docs) == 1  # single Mongo doc
    stored = db.colls["chunk_embeddings"].find_one({"course_id": "course-1"})
    assert stored["chunks"] == ["c1", "c2"]
    assert (np.asarray(stored["embeddings"], dtype="float32") == first).all()