"""EvaluatorWorker unit tests (F04 / EVAL-01) — no broker.

A real EvaluatorAgent is injected with ``require_llm=False`` so the multi-turn
Socratic state machine is exercised deterministically through the worker. The
tests pin: routing start-vs-resume, terminal rejection of malformed input,
off-loop execution and ACK/NACK through the BaseAIWorker pipeline.
"""

from __future__ import annotations

import json
import uuid

import pytest

from messaging.failures import TerminalError
from workers.base import BaseAIWorker
from workers.evaluator_worker import EvaluatorWorker
from workers.idempotency import InMemoryIdempotencyStore


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
        "type": "study.eval.step",
        "version": "1",
        "userId": "user-42",
        "requestId": "req-1",
        "timestamp": "2026-08-26T08:00:00Z",
        "payload": payload,
    }


def make_worker(agent=None):
    from agents.evaluator.agent import EvaluatorAgent

    real = agent if agent is not None else EvaluatorAgent(require_llm=False)

    class RecordingWorker(EvaluatorWorker):
        def __init__(self):
            super().__init__(idempotency_store=InMemoryIdempotencyStore(), agent=real)
            self.results = []

        async def _publish_result(self, env, *, status, payload=None, error=None):
            self.results.append((status, payload, error))

    return RecordingWorker()


async def consume(worker, message):
    await worker.on_message(message)


# ---------------------------------------------------------------- start session

async def test_start_session_payload_publishes_completed_result():
    worker = make_worker()
    msg = FakeMessage(envelope({
        "task_title": "Photosynthesis",
        "task_description": "plants convert sunlight to energy",
        "task_details": "Light reactions produce ATP and NADPH. Calvin cycle fixes CO2 into glucose.",
        "max_attempts": 5,
    }))
    await consume(worker, msg)

    assert msg.acked and not msg.nacked
    status, payload, error = worker.results[0]
    assert status == "completed"
    assert error is None
    assert payload["session_id"]
    assert payload["question"]


# --------------------------------------------------------- resume a session step

async def test_resume_session_publishes_completed_result():
    worker = make_worker()
    start_msg = FakeMessage(envelope({
        "task_title": "Python Lists",
        "task_description": "how lists work",
        "task_details": "Lists are ordered mutable sequences supporting indexing, slicing, append, remove, pop.",
        "max_attempts": 5,
    }))
    await consume(worker, start_msg)
    sid = worker.results[0][1]["session_id"]

    worker.results = []
    resume_msg = FakeMessage(envelope({
        "session_id": sid,
        "student_answer": "Python lists are ordered collections created with square brackets that support append and pop.",
    }))
    await consume(worker, resume_msg)

    assert resume_msg.acked and not resume_msg.nacked
    status, payload, error = worker.results[0]
    assert status == "completed"
    assert error is None
    assert payload["session_id"] == sid
    assert payload["state"] in ("MASTERY_CONFIRMED", "CONTINUE", "FAILED")
    assert 0.0 <= payload["mastery_score"] <= 1.0


# ----------------------------------------------------------- invalid input

async def test_non_object_payload_is_terminal():
    # A non-object payload fails envelope validation in BaseAIWorker before the
    # worker's handle() runs → TerminalError is raised (no result event).
    worker = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope("just a string")))
    assert worker.results == []


async def test_empty_payload_is_terminal():
    worker = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({})))
    assert worker.results[0][0] == "failed"


async def test_resume_without_student_answer_is_terminal():
    worker = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"session_id": "abc"})))
    assert worker.results[0][0] == "failed"


async def test_start_without_task_details_is_terminal():
    worker = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"task_title": "t"})))
    assert worker.results[0][0] == "failed"


# ------------------------------------------------------------ retry policy

async def test_transient_agent_failure_schedules_retry():
    class FailingAgent:
        def start_session(self, **kwargs):
            raise RuntimeError("LLM timeout")

    worker = make_worker(agent=FailingAgent())
    msg = FakeMessage(envelope({
        "task_title": "t", "task_description": "d",
        "task_details": "some details", "max_attempts": 5,
    }), headers={"x-retry-count": 0})

    published = {}

    async def fake_republish(message, env, next_attempt, delay_ms, reason):
        published["next"] = next_attempt
        message.ack()

    worker._republish_for_retry = fake_republish
    await consume(worker, msg)

    # RuntimeError is not Terminal/Retryable → classified retryable, republished
    assert published["next"] == 1
    assert msg.acked
    assert worker.results == []


# ------------------------------------------------------------ wiring / laziness

async def test_agent_runs_off_event_loop_via_to_thread():
    import threading

    from agents.evaluator.agent import EvaluatorAgent

    real = EvaluatorAgent(require_llm=False)
    seen = []
    original = real.start_session

    def recording_start(**kwargs):
        seen.append(threading.current_thread())
        return original(**kwargs)

    real.start_session = recording_start
    worker = make_worker(agent=real)
    await consume(worker, FakeMessage(envelope({
        "task_title": "t", "task_description": "d",
        "task_details": "some details for thread check",
    })))
    assert seen[0] is not threading.main_thread()


def test_lazy_agent_not_built_until_first_handle():
    built = {"count": 0}

    class CountingWorker(EvaluatorWorker):
        @property
        def agent(self):
            if self._agent is None:
                built["count"] += 1
                self._agent = object()
            return self._agent

    w = CountingWorker(idempotency_store=InMemoryIdempotencyStore())
    assert built["count"] == 0


def test_job_type_and_base_class():
    assert EvaluatorWorker.job_type == "study.eval.step"
    assert issubclass(EvaluatorWorker, BaseAIWorker)
