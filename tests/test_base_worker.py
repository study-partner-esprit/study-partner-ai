"""BaseAIWorker unit tests (AI-COM-05/06/08) — no broker required.

The broker-facing pieces (_publish_result, _republish_for_retry, message
context manager) are faked; the tests pin the ACK/NACK pipeline decisions:
dispatch, idempotency, retry scheduling and terminal dead-lettering.
"""

from __future__ import annotations

import json
import uuid

import pytest

from messaging.failures import RetryableError, TerminalError, classify_failure
from workers.base import BaseAIWorker, _RetryScheduled
from workers.idempotency import InMemoryIdempotencyStore


class FakeMessage:
    def __init__(self, envelope: dict, headers: dict | None = None):
        self.body = json.dumps(envelope).encode()
        self.headers = headers or {}
        self.acked = False
        self.nacked = False
        self.requeue: bool | None = None

    def ack(self):
        assert not self.acked and not self.nacked
        self.acked = True

    def nack(self, multiple=False, requeue=False):
        assert not self.acked and not self.nacked
        self.nacked = True
        self.requeue = requeue

    class _Ctx:
        def __init__(self, msg):
            self.msg = msg

        async def __aenter__(self):
            return self.msg

        async def __aexit__(self, exc_type, exc, tb):
            if self.msg.acked or self.msg.nacked:
                return False  # already processed; ignore_processed semantics
            if exc_type is None:
                self.msg.ack()
            else:
                self.msg.nack(requeue=False)
            return False  # propagate exceptions like aio_pika would log them

    def process(self, ignore_processed=True):
        return FakeMessage._Ctx(self)


def envelope(**overrides) -> dict:
    base = {
        "messageId": str(uuid.uuid4()),
        "correlationId": "0f8e2d1a-3b4c-4d6e-8f80-91a2b3c4d5e6",
        "type": "study.plan.generate",
        "version": "1",
        "userId": "user-1",
        "requestId": "req-1",
        "timestamp": "2026-08-19T08:00:00Z",
        "payload": {"goal": "learn"},
    }
    base.update(overrides)
    return base


class StubWorker(BaseAIWorker):
    job_type = "study.plan.generate"

    def __init__(self, behaviour=None):
        super().__init__(idempotency_store=InMemoryIdempotencyStore())
        self.behaviour = behaviour or (lambda payload, env: {"ok": True})
        self.calls = []
        self.results = []
        self.retries = []

    async def handle(self, payload, env):
        self.calls.append(payload)
        return self.behaviour(payload, env)

    async def _publish_result(self, env, *, status, payload=None, error=None):
        self.results.append((status, payload, error))


@pytest.fixture
def worker():
    return StubWorker()


async def consume(worker, message):
    await worker.on_message(message)


# --------------------------------------------------------------- happy path

async def test_success_publishes_completed_result_and_acks(worker):
    await consume(worker, FakeMessage(envelope()))
    assert len(worker.calls) == 1
    assert worker.results == [("completed", {"ok": True}, None)]


# -------------------------------------------------------------- idempotency

async def test_duplicate_message_acknowledged_without_rerun(worker):
    msg = FakeMessage(envelope())
    await consume(worker, msg)
    duplicate = FakeMessage(json.loads(msg.body))
    await consume(worker, duplicate)
    assert len(worker.calls) == 1
    assert duplicate.acked and not duplicate.nacked


# ----------------------------------------------------------------- retries

async def test_retryable_failure_republishes_to_first_delay(worker):
    worker.behaviour = lambda p, e: (_ for _ in ()).throw(RetryableError("LLM timeout"))
    msg = FakeMessage(envelope(), headers={"x-retry-count": 0})

    published = {}

    async def fake_republish(message, env, next_attempt, delay_ms, reason):
        published["next"] = next_attempt
        published["delay"] = delay_ms
        message.ack()
        raise _RetryScheduled()

    worker._republish_for_retry = fake_republish
    with pytest.raises(_RetryScheduled):
        await consume(worker, msg)

    assert published == {"next": 1, "delay": 1000}
    assert msg.acked
    assert worker.results == []  # no failure result until retries exhausted


async def test_exhausted_retries_dead_letter_with_failed_result(worker):
    worker.behaviour = lambda p, e: (_ for _ in ()).throw(RetryableError("still down"))
    msg = FakeMessage(envelope(), headers={"x-retry-count": 3})  # MAX_RETRIES reached

    with pytest.raises(TerminalError):
        await consume(worker, msg)

    assert worker.results[0][0] == "failed"
    assert "still down" in worker.results[0][2]
    assert msg.nacked and msg.requeue is False  # DLX → ai.dlq.<type>


async def test_terminal_failure_skips_retries_entirely(worker):
    worker.behaviour = lambda p, e: (_ for _ in ()).throw(ValueError("invalid payload"))
    msg = FakeMessage(envelope())

    with pytest.raises(TerminalError):
        await consume(worker, msg)

    assert worker.results[0][0] == "failed"
    assert msg.nacked and msg.requeue is False


# ------------------------------------------------------------ invalid input

async def test_invalid_envelope_dead_lettered_without_dispatch(worker):
    bad = envelope(type="not.a.type")
    msg = FakeMessage(bad)

    with pytest.raises(TerminalError):
        await consume(worker, msg)

    assert worker.calls == []
    assert msg.nacked and msg.requeue is False


# --------------------------------------------------------- classification

def test_classification_matches_node_semantics():
    from messaging.failures import FailureClass as FC

    retryable = [
        TimeoutError("request timeout"),
        ConnectionResetError("ECONNRESET"),
        RuntimeError("HTTP 503 temporarily unavailable"),
        RuntimeError("rate limit exceeded"),
        Exception("quota exceeded"),
    ]
    terminal = [
        ValueError("validation failed"),
        KeyError("invalid payload"),
        PermissionError("unauthorized"),
        LookupError("job rejected by validator"),
    ]
    for err in retryable:
        assert classify_failure(err) is FC.RETRYABLE, err
    for err in terminal:
        assert classify_failure(err) is FC.TERMINAL, err
