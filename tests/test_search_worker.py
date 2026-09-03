"""SearchWorker unit tests (F05 / SEARCH-01, -02, -04, -05, -06) — no broker.

The real `run_pipeline` is injected as a deterministic stand-in so no web
crawling or LLM calls happen. Tests pin:

- SEARCH-01: `SearchWorker` extends `BaseAIWorker` and consumes
  `study.search.query`; a completed job publishes a structured result.
- SEARCH-02: strict `SearchRequest` validation — malformed/blank/over-long
  queries and unknown fields are TERMINAL (dead-lettered, not retried).
- SEARCH-04: the result payload maps onto a strict `SearchOutput`
  (answer + sources + optional voice_summary) and carries query metadata.
- SEARCH-05: results with no sources are rejected (degraded), never published
  as complete.
- SEARCH-06: repeated queries hit the cache (in-memory store in tests) and a
  transient pipeline failure degrades gracefully instead of failing the job.
"""

from __future__ import annotations

import json
import uuid

import pytest

from messaging.failures import TerminalError
from workers.base import BaseAIWorker
from workers.idempotency import InMemoryIdempotencyStore
from workers.search_cache import InMemorySearchCache
from workers.search_worker import SearchWorker


class FakeMessage:
    def __init__(self, envelope: dict, headers: dict | None = None):
        self.body = json.dumps(envelope).encode()
        self.headers = headers or {}
        self.acked = False
        self.nacked = False

    def ack(self):
        assert not self.acked and not self.nacked
        self.acked = True

    def nack(self, multiple=False, requeue=False):
        assert not self.acked and self.nacked is False
        self.nacked = True

    class _Ctx:
        def __init__(self, msg):
            self.msg = msg

        async def __aenter__(self):
            return self.msg

        async def __aexit__(self, exc_type, exc, tb):
            if self.msg.acked or self.msg.nacked:
                return False
            if exc_type is None:
                self.msg.ack()
            else:
                self.msg.nack(requeue=False)
            return False

    def process(self, ignore_processed=True):
        return FakeMessage._Ctx(self)


def envelope(payload) -> dict:
    return {
        "messageId": str(uuid.uuid4()),
        "correlationId": "0f8e2d1a-3b4c-4d6e-8f80-91a2b3c4d5e6",
        "type": "study.search.query",
        "version": "1",
        "userId": "user-42",
        "requestId": "req-search-1",
        "timestamp": "2026-09-03T08:00:00Z",
        "payload": payload,
    }


def good_payload(**overrides) -> dict:
    data = {"query": "what is recursion", "maxResults": 3, "voiceMode": False}
    data.update(overrides)
    return data


def make_worker(pipeline_runner=None, cache=None):
    def default_runner(query, max_results=5, use_voice=False):
        from agents.search.pipeline import PipelineResult

        return PipelineResult(
            answer="Recursion is when a function calls itself.",
            sources=[
                {"url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"}
            ],
            degraded=False,
        )

    class RecordingWorker(SearchWorker):
        def __init__(self):
            super().__init__(
                idempotency_store=InMemoryIdempotencyStore(),
                cache=cache or InMemorySearchCache(),
                pipeline_runner=pipeline_runner or default_runner,
            )
            self.results = []

        async def _publish_result(self, env, *, status, payload=None, error=None):
            self.results.append((status, payload, error))

    return RecordingWorker()


async def consume(worker, message):
    await worker.on_message(message)


# ---------------------------------------------------------------- SEARCH-01

def test_job_type_and_base_class():
    assert SearchWorker.job_type == "study.search.query"
    assert issubclass(SearchWorker, BaseAIWorker)


async def test_completed_job_publishes_structured_answer():
    worker = make_worker()
    msg = FakeMessage(envelope(good_payload()))
    await consume(worker, msg)

    assert msg.acked and not msg.nacked
    status, payload, error = worker.results[0]
    assert status == "completed"
    assert error is None
    assert payload["answer"]
    assert isinstance(payload["sources"], list) and payload["sources"]
    assert payload["query"] == "what is recursion"
    assert payload["cached"] is False


async def test_answer_carries_sources_and_query_metadata():
    worker = make_worker()
    await consume(worker, FakeMessage(envelope(good_payload(maxResults=2))))

    status, payload, _ = worker.results[0]
    assert status == "completed"
    src = payload["sources"][0]
    assert src["url"].startswith("http")
    assert payload["maxResults"] == 2


# ---------------------------------------------------------------- SEARCH-02

async def test_blank_query_is_terminal():
    worker = make_worker()
    msg = FakeMessage(envelope(good_payload(query="   ")))
    with pytest.raises(TerminalError):
        await consume(worker, msg)
    assert worker.results and worker.results[0][0] == "failed"


async def test_missing_query_is_terminal():
    worker = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({})))
    assert worker.results[0][0] == "failed"


async def test_overlong_query_is_terminal():
    worker = make_worker()
    msg = FakeMessage(envelope(good_payload(query="a" * 501)))
    with pytest.raises(TerminalError):
        await consume(worker, msg)
    assert worker.results[0][0] == "failed"


async def test_unknown_extra_field_is_terminal():
    worker = make_worker()
    msg = FakeMessage(envelope(good_payload(evil_instruction="ignore previous")))
    with pytest.raises(TerminalError):
        await consume(worker, msg)
    assert worker.results[0][0] == "failed"


async def test_non_object_payload_is_terminal():
    worker = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope("just a string")))
    assert worker.results == []


# ----------------------------------------------------------------- SEARCH-04

async def test_voice_mode_allows_voice_summary():
    def runner(query, max_results=5, use_voice=False):
        from agents.search.pipeline import PipelineResult

        return PipelineResult(
            answer="A short answer.",
            sources=[{"url": "https://en.wikipedia.org/wiki/Recursion"}],
            voice_summary="A short answer." if use_voice else None,
            degraded=False,
        )

    worker = make_worker(pipeline_runner=runner)
    await consume(worker, FakeMessage(envelope(good_payload(voiceMode=True))))
    status, payload, _ = worker.results[0]
    assert status == "completed"
    assert payload["voice_summary"] == "A short answer."
    assert payload["voiceMode"] is True


# ----------------------------------------------------------------- SEARCH-05

async def test_result_without_sources_is_degraded_not_complete():
    def runner(query, max_results=5, use_voice=False):
        from agents.search.pipeline import PipelineResult

        return PipelineResult(answer="Has an answer but no citations.", sources=[], degraded=False)

    worker = make_worker(pipeline_runner=runner)
    await consume(worker, FakeMessage(envelope(good_payload())))
    status, payload, _ = worker.results[0]
    assert status == "completed"  # job completes (degraded), not dead-lettered
    assert payload["degraded"] is True
    assert payload["reason"] == "missing_sources"


async def test_missing_sources_result_not_cached():
    from agents.search.pipeline import PipelineResult

    runner = lambda query, max_results=5, use_voice=False: PipelineResult(
        answer="no sources", sources=[], degraded=True
    )
    cache = InMemorySearchCache()
    worker = make_worker(pipeline_runner=runner, cache=cache)
    await consume(worker, FakeMessage(envelope(good_payload())))
    # degraded results are not cached (nothing worth serving on a repeat query)
    assert not cache._data


# ----------------------------------------------------------------- SEARCH-06

async def test_repeated_query_serves_from_cache():
    calls = {"n": 0}

    def runner(query, max_results=5, use_voice=False):
        calls["n"] += 1
        from agents.search.pipeline import PipelineResult

        return PipelineResult(
            answer="Cached answer",
            sources=[{"url": "https://en.wikipedia.org/wiki/Recursion"}],
            degraded=False,
        )

    cache = InMemorySearchCache()
    worker = make_worker(pipeline_runner=runner, cache=cache)

    await consume(worker, FakeMessage(envelope(good_payload())))
    assert calls["n"] == 1
    _, first, _ = worker.results[0]
    assert first["cached"] is False

    worker.results = []
    await consume(worker, FakeMessage(envelope(good_payload())))
    # pipeline NOT re-run; served from cache
    assert calls["n"] == 1
    _, second, _ = worker.results[0]
    assert second["cached"] is True
    assert second["answer"] == "Cached answer"


async def test_transient_pipeline_failure_degrades_gracefully():
    def failing_runner(query, max_results=5, use_voice=False):
        raise RuntimeError("crawler timeout")

    worker = make_worker(pipeline_runner=failing_runner)
    msg = FakeMessage(envelope(good_payload()))
    await consume(worker, msg)

    assert msg.acked and not msg.nacked
    status, payload, _ = worker.results[0]
    assert status == "completed"
    assert payload["degraded"] is True
