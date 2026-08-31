"""EvaluatorWorker unit tests (F04 / EVAL-01, EVAL-02) — no broker.

A real EvaluatorAgent is injected with ``require_llm=False`` so the multi-turn
Socratic state machine is exercised deterministically through the worker. Tests
pin: step-driven routing (step 1 starts, step > 1 resumes), strict
EvaluationRequest validation, session rehydration from the state store,
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
from workers.session_store import InMemorySessionStore


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


DEFAULT_CONTEXT = {
    "task_title": "Task Title",
    "task_description": "Task description",
    "task_details": (
        "Recursion requires a base case that stops the recursion and a recursive "
        "case that reduces the problem toward the base case using the call stack."
    ),
}


def resolver(context_id):
    ctx = dict(DEFAULT_CONTEXT)
    ctx["task_title"] = f"{ctx['task_title']} ({context_id})"
    return ctx


def make_worker(agent=None, session_store=None):
    from agents.evaluator.agent import EvaluatorAgent

    real = agent if agent is not None else EvaluatorAgent(require_llm=False)

    class RecordingWorker(EvaluatorWorker):
        def __init__(self):
            super().__init__(
                idempotency_store=InMemoryIdempotencyStore(),
                session_store=session_store or InMemorySessionStore(),
                agent=real,
                context_resolver=resolver,
            )
            self.results = []

        async def _publish_result(self, env, *, status, payload=None, error=None):
            self.results.append((status, payload, error))

    return RecordingWorker()


async def consume(worker, message):
    await worker.on_message(message)


def start_payload(session_id="sess-1", answer="Recursion needs a base case so it stops."):
    return {
        "sessionId": session_id,
        "step": 1,
        "contextId": "ctx-recursion",
        "studentAnswer": answer,
    }


# ---------------------------------------------------------------- start session

async def test_step1_starts_and_processes_first_answer():
    worker = make_worker()
    msg = FakeMessage(envelope(start_payload("sess-start")))
    await consume(worker, msg)

    assert msg.acked and not msg.nacked
    status, payload, error = worker.results[0]
    assert status == "completed"
    assert error is None
    assert payload["session_id"] == "sess-start"
    assert 0.0 <= payload["mastery_score"] <= 1.0
    assert payload["state"] in ("MASTERY_CONFIRMED", "CONTINUE", "FAILED")
    # the session was persisted to the state store
    assert await worker.session_store.get("sess-start") is not None


# --------------------------------------------------------- resume a session step

async def test_step2_resumes_session_step():
    worker = make_worker()
    await consume(worker, FakeMessage(envelope(start_payload("sess-resume"))))
    sid = "sess-resume"

    worker.results = []
    resume_msg = FakeMessage(envelope({
        "sessionId": sid,
        "step": 2,
        "contextId": "ctx-recursion",
        "studentAnswer": "A base case stops recursion, recursive case reduces the problem.",
    }))
    await consume(worker, resume_msg)

    assert resume_msg.acked and not resume_msg.nacked
    status, payload, error = worker.results[0]
    assert status == "completed"
    assert error is None
    assert payload["session_id"] == sid
    assert payload["state"] in ("MASTERY_CONFIRMED", "CONTINUE", "FAILED", "complete")


# ------------------------------------------------------- session rehydration

async def test_resume_rehydrates_session_into_fresh_agent():
    # A session is started by worker A and persisted to a SHARED store. Worker B
    # (a fresh agent that never held the session) resumes step 2 → hydrated from
    # the shared state store (mirrors cross-instance rehydration via Redis).
    shared_store = InMemorySessionStore()
    worker_a = make_worker(session_store=shared_store)
    await consume(worker_a, FakeMessage(envelope(start_payload("sess-rehyd"))))
    serialized = await shared_store.get("sess-rehyd")
    assert serialized is not None

    worker_b = make_worker(session_store=shared_store)
    assert worker_b.agent.get_session("sess-rehyd") is None

    resume_msg = FakeMessage(envelope({
        "sessionId": "sess-rehyd",
        "step": 2,
        "contextId": "ctx-recursion",
        "studentAnswer": "It reduces toward the base case using the call stack.",
    }))
    await consume(worker_b, resume_msg)

    # The session was rehydrated into worker B's agent (restore_session ran)
    assert worker_b.agent.get_session("sess-rehyd") is not None
    assert resume_msg.acked and not resume_msg.nacked
    status, payload, error = worker_b.results[0]
    assert status == "completed"
    assert error is None
    assert payload["session_id"] == "sess-rehyd"


async def test_resume_without_store_entry_returns_session_not_found():
    worker = make_worker()
    msg = FakeMessage(envelope({
        "sessionId": "sess-unknown",
        "step": 2,
        "contextId": "ctx-recursion",
        "studentAnswer": "some answer",
    }))
    await consume(worker, msg)

    assert msg.acked and not msg.nacked
    status, payload, error = worker.results[0]
    assert status == "completed"
    assert payload.get("error") == "session_not_found"


# ----------------------------------------------------------- invalid input

async def test_non_object_payload_is_terminal():
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
        await consume(worker, FakeMessage(envelope({
            "sessionId": "abc", "step": 2, "contextId": "ctx",
        })))
    assert worker.results[0][0] == "failed"


async def test_unknown_extra_field_is_terminal():
    worker = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({
            "sessionId": "abc", "step": 2, "contextId": "ctx",
            "studentAnswer": "ok", "evil_instruction": "ignore previous",
        })))
    assert worker.results[0][0] == "failed"


# ------------------------------------------------------------ retry policy

async def test_transient_agent_failure_schedules_retry():
    class FailingAgent:
        def start_session(self, **kwargs):
            raise RuntimeError("LLM timeout")

    worker = make_worker(agent=FailingAgent())
    msg = FakeMessage(envelope(start_payload("sess-fail")), headers={"x-retry-count": 0})

    published = {}

    async def fake_republish(message, env, next_attempt, delay_ms, reason):
        published["next"] = next_attempt
        message.ack()

    worker._republish_for_retry = fake_republish
    await consume(worker, msg)

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
    await consume(worker, FakeMessage(envelope(start_payload("sess-thread"))))
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

    w = CountingWorker(
        idempotency_store=InMemoryIdempotencyStore(),
        session_store=InMemorySessionStore(),
    )
    assert built["count"] == 0


def test_job_type_and_base_class():
    assert EvaluatorWorker.job_type == "study.eval.step"
    assert issubclass(EvaluatorWorker, BaseAIWorker)
