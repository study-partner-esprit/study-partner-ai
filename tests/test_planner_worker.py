"""PlannerWorker unit tests (F02 / PLAN-01) — no broker, no heavy agent.

The PlannerAgent is faked; the tests pin the worker's contract:
payload→PlannerInput mapping, terminal rejection of malformed input,
off-loop execution and result publishing through the BaseAIWorker pipeline.
"""

from __future__ import annotations

import json
import uuid

import pytest

from messaging.failures import RetryableError, TerminalError
from workers.base import BaseAIWorker
from workers.idempotency import InMemoryIdempotencyStore
from workers.planner_worker import DEFAULT_AVAILABLE_MINUTES, PlannerWorker


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
        "type": "study.plan.generate",
        "version": "1",
        "userId": "user-42",
        "requestId": "req-1",
        "timestamp": "2026-08-26T08:00:00Z",
        "payload": payload,
    }


class FakePlannerAgent:
    """Stands in for PlannerAgent; records requests, returns canned output."""

    def __init__(self, behaviour=None):
        self.requests = []
        self.behaviour = behaviour or self._default

    @staticmethod
    def _default(request):
        from agents.planner.models.task_graph import AtomicTask, PlannerOutput, TaskGraph

        # Return at least one task so output validation (PLAN-05) passes
        task = AtomicTask(
            id="task-0",
            title="Study the material",
            description="Review and practice",
            estimated_minutes=min(request.available_minutes, 45),
            difficulty=0.5,
            prerequisites=[],
            is_review=False,
        )
        return PlannerOutput(
            task_graph=TaskGraph(goal=request.goal or "", tasks=[task]),
            warning=None,
            clarification_required=False,
        )

    def plan(self, request):
        self.requests.append(request)
        result = self.behaviour(request)
        if isinstance(result, Exception):
            raise result
        return result


def make_worker(behaviour=None):
    agent = FakePlannerAgent(behaviour)

    class RecordingWorker(PlannerWorker):
        def __init__(self):
            super().__init__(idempotency_store=InMemoryIdempotencyStore(), agent=agent)
            self.results = []

        async def _publish_result(self, env, *, status, payload=None, error=None):
            self.results.append((status, payload, error))

    return RecordingWorker(), agent


async def consume(worker, message):
    await worker.on_message(message)


# ---------------------------------------------------------------- happy path

async def test_valid_goal_payload_publishes_completed_result():
    worker, agent = make_worker()
    msg = FakeMessage(
        envelope({"goal": "learn rabbitmq", "available_minutes": 90, "concepts": ["exchanges"]})
    )

    await consume(worker, msg)

    assert msg.acked and not msg.nacked
    assert len(agent.requests) == 1
    req = agent.requests[0]
    assert req.goal == "learn rabbitmq"
    assert req.available_minutes == 90
    assert req.user_id == "user-42"  # identity from envelope, never payload
    assert req.retrieved_concepts == ["exchanges"]
    status, payload, error = worker.results[0]
    assert status == "completed"
    assert error is None
    assert payload["task_graph"]["goal"] == "learn rabbitmq"


async def test_defaults_fill_missing_optional_fields():
    worker, agent = make_worker()
    await consume(worker, FakeMessage(envelope({"goal": "solo goal"})))

    req = agent.requests[0]
    assert req.available_minutes == DEFAULT_AVAILABLE_MINUTES
    assert req.deadline_iso  # defaulted ISO deadline present


async def test_agent_runs_off_event_loop_via_to_thread(monkeypatch):
    worker, agent = make_worker()
    seen_threads = []

    import threading

    real_plan = agent.plan

    def plan_records_thread(request):
        seen_threads.append(threading.current_thread())
        return real_plan(request)

    agent.plan = plan_records_thread

    await consume(worker, FakeMessage(envelope({"goal": "thread check"})))
    # to_thread executes in a worker thread, not the main loop thread
    assert seen_threads[0] is not threading.main_thread()


# ----------------------------------------------------------- invalid input

async def test_empty_payload_is_terminal_without_retry():
    worker, agent = make_worker()
    msg = FakeMessage(envelope({}))

    with pytest.raises(TerminalError):
        await consume(worker, msg)

    assert msg.nacked and not msg.acked
    assert agent.requests == []
    # failed result published for correlation (AiJob → FAILED)
    assert worker.results[0][0] == "failed"


async def test_non_integer_minutes_is_terminal():
    worker, _ = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"goal": "g", "available_minutes": "lots"})))


async def test_non_object_payload_is_terminal():
    worker, _ = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope("just a string")))


async def test_planner_input_validation_failure_maps_to_terminal():
    # Force PlannerInput construction failure (deadline wrong type)
    worker, _ = make_worker()

    bad = envelope({"goal": "g", "deadline_iso": 12345, "available_minutes": 30})
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(bad))


# ------------------------------------------------------------ retry policy

async def test_transient_agent_failure_schedules_retry():
    worker, agent = make_worker(lambda request: (_ for _ in ()).throw(RetryableError("LLM timeout")))
    msg = FakeMessage(envelope({"goal": "retry me"}), headers={"x-retry-count": 0})

    published = {}

    async def fake_republish(message, env, next_attempt, delay_ms, reason):
        published["next"] = next_attempt
        message.ack()

    worker._republish_for_retry = fake_republish
    await consume(worker, msg)

    assert published["next"] == 1
    assert msg.acked
    assert worker.results == []  # no terminal result while retries remain


async def test_lazy_agent_not_built_until_first_handle():
    built = {"count": 0}

    class CountingWorker(PlannerWorker):
        @property
        def agent(self):
            if self._agent is None:
                built["count"] += 1
                self._agent = object()  # would explode if used as real agent
            return self._agent

    w = CountingWorker(idempotency_store=InMemoryIdempotencyStore())
    assert built["count"] == 0  # constructor did not touch the agent
