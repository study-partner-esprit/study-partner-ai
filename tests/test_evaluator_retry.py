"""EVAL-07 — retry-safe session state (validate-then-mutate + RetryableError).

Pins that a transient LLM failure (timeout/quota/5xx) surfaces as a
``RetryableError`` BEFORE any session mutation, so the worker can retry the
step without double-counting attempts or answers (state machine is
retry-safe). Non-transient failures (validation / mock-missing) keep the
defensive local-scoring fallback. Depends on the shared AI-COM-06 retry/DLQ
policy at the worker layer.
"""

from __future__ import annotations

import json
import uuid

import pytest

from messaging.failures import FailureClass, RetryableError, classify_failure
from utils.llm_client import LLMRequestError, MissingMockResponderError

from agents.evaluator.agent import EvaluatorAgent
from agents.evaluator.schemas import SessionState

# A valid follow-up "analysis" line the validator accepts. It must include at
# least one of the task's key concepts in the answer to pass grounding; we keep
# analysis responses simple and rely on the fake LLM.
TASK = {
    "task_title": "Recursion",
    "task_description": "Understand recursion",
    "task_details": (
        "Recursion requires a base case that stops the recursion and a recursive "
        "case that reduces the problem toward the base case."
    ),
}


class FakeLLM:
    """Injected evaluator LLM with scripted behavior per call."""

    def __init__(self, generate_impl=None, question="What is a base case in recursion?"):
        self._generate = generate_impl or (lambda prompt, max_tokens, raise_on_error: "ok")
        self.question = question
        self.analysis_calls = 0

    def generate_question(self, prompt, max_tokens=200, temperature=0.3, **kwargs):
        return self.question

    def generate(self, prompt, max_tokens=200, temperature=0.3, raise_on_error=False):
        self.analysis_calls += 1
        return self._generate(prompt, max_tokens, raise_on_error)


def make_agent(llm) -> EvaluatorAgent:
    return EvaluatorAgent(llm_client=llm)


def start_session(agent: EvaluatorAgent, session_id="sess-retry") -> str:
    agent.start_session(
        task_title=TASK["task_title"],
        task_description=TASK["task_description"],
        task_details=TASK["task_details"],
        session_id=session_id,
    )
    return session_id


# ------------------------------------------------- transient failure -> RetryableError

def test_transient_timeout_raises_retryable_and_does_not_mutate_state():
    def impl(prompt, max_tokens, raise_on_error):
        raise LLMRequestError("evaluator: anthropic timeout after 60s")

    agent = make_agent(FakeLLM(generate_impl=impl))
    sid = start_session(agent)

    with pytest.raises(RetryableError):
        agent.handle_user_answer(sid, "A base case stops the recursion.")

    session = agent.get_session(sid)
    # No state advanced: the step never "counted".
    assert session.attempts == 0
    assert session.answer_history == []
    assert session.state == SessionState.ASKING


def test_transient_quota_raises_retryable_no_mutation():
    def impl(prompt, max_tokens, raise_on_error):
        raise LLMRequestError("evaluator: 429 quota exceeded")

    agent = make_agent(FakeLLM(generate_impl=impl))
    sid = start_session(agent)
    with pytest.raises(RetryableError):
        agent.handle_user_answer(sid, "A base case stops the recursion.")
    session = agent.get_session(sid)
    assert session.attempts == 0
    assert session.answer_history == []


def test_retryable_raised_only_when_transient():
    # Non-transient (validation/parse) failure must NOT raise RetryableError —
    # it falls back to local scoring so the session continues.
    def impl(prompt, max_tokens, raise_on_error):
        raise LLMRequestError("evaluator: malformed schema in response")

    agent = make_agent(FakeLLM(generate_impl=impl))
    sid = start_session(agent)
    result = agent.handle_user_answer(sid, "A base case stops the recursion.")
    session = agent.get_session(sid)
    assert session.attempts == 1  # committed
    assert session.answer_history == ["A base case stops the recursion."]
    assert 0.0 <= result["mastery_score"] <= 1.0


def test_missing_mock_failure_falls_back_not_retryable():
    def impl(prompt, max_tokens, raise_on_error):
        raise MissingMockResponderError("no API key for 'evaluator' and no mock_fn")

    agent = make_agent(FakeLLM(generate_impl=impl))
    sid = start_session(agent)
    result = agent.handle_user_answer(sid, "A base case stops the recursion.")
    assert result["mastery_score"] >= 0.0
    assert agent.get_session(sid).attempts == 1


# -------------------------------------------------- retry does not double-count

def test_retry_replay_does_not_double_count():
    # First call transient → RetryableError (nothing counted). The worker then
    # re-delivers; the retry succeeds. Attempts must be exactly 1.
    calls = {"n": 0}

    def impl(prompt, max_tokens, raise_on_error):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMRequestError("evaluator: connection reset by peer")
        return "The base case stops the recursion."

    agent = make_agent(FakeLLM(generate_impl=impl))
    sid = start_session(agent)

    with pytest.raises(RetryableError):
        agent.handle_user_answer(sid, "A base case stops the recursion.")

    result = agent.handle_user_answer(sid, "A base case stops the recursion.")
    session = agent.get_session(sid)

    assert session.attempts == 1  # not 2
    assert session.answer_history == ["A base case stops the recursion."]
    assert result["state"] in ("MASTERY_CONFIRMED", "CONTINUE", "FAILED")


# --------------------------------------------------------- failure classification

def test_llm_error_classified_retryable_for_timeout():
    err = RuntimeError("evaluator: request timed out")
    assert classify_failure(err) == FailureClass.RETRYABLE


def test_llm_error_classified_terminal_for_validation():
    err = RuntimeError("evaluator: invalid schema")
    assert classify_failure(err) == FailureClass.TERMINAL


# ------------------------------------------------- client-level raise_on_error flag

def test_generate_propagates_transient_when_raise_on_error_true(monkeypatch):
    import agents.evaluator.llm_client as llm_client_mod
    from agents.evaluator.llm_client import GeminiClient

    def fake_ask(agent, system_prompt, user_prompt, **kwargs):
        raise LLMRequestError("evaluator: service unavailable 503")

    monkeypatch.setattr(llm_client_mod, "ask", fake_ask)
    client = GeminiClient()
    with pytest.raises(LLMRequestError):
        client.generate("prompt", raise_on_error=True)


def test_generate_degrades_to_placeholder_when_raise_on_error_false(monkeypatch):
    import agents.evaluator.llm_client as llm_client_mod
    from agents.evaluator.llm_client import GeminiClient

    def fake_ask(agent, system_prompt, user_prompt, **kwargs):
        raise LLMRequestError("evaluator: service unavailable 503")

    monkeypatch.setattr(llm_client_mod, "ask", fake_ask)
    client = GeminiClient()
    out = client.generate("prompt", raise_on_error=False)
    assert "unavailable" in out.lower()


# ------------------------------------------------------------------ worker level
# The full ACK/NACK + retry pipeline (AI-COM-06) is exercised through the real
# EvaluatorWorker + BaseAIWorker with a fake broker message.

def _envelope(payload) -> dict:
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


def _make_worker(agent):
    from workers.evaluator_worker import EvaluatorWorker
    from workers.idempotency import InMemoryIdempotencyStore
    from workers.session_store import InMemorySessionStore

    class RecordingWorker(EvaluatorWorker):
        def __init__(self):
            super().__init__(
                idempotency_store=InMemoryIdempotencyStore(),
                session_store=InMemorySessionStore(),
                agent=agent,
            )
            self.results = []

        async def _publish_result(self, env, *, status, payload=None, error=None):
            self.results.append((status, payload, error))

    return RecordingWorker()


async def test_worker_retryable_analysis_failure_schedules_retry():
    # A transient (timeout/quota) analysis failure surfaces as RetryableError,
    # so the worker schedules a retry (AI-COM-06) rather than dead-lettering.
    # Because the evaluator validates-then-mutates, the session state is left
    # untouched — a retry replays without double-counting.
    from agents.evaluator.agent import EvaluatorAgent

    class FailFirstThenOkLLM:
        def __init__(self):
            self.q = "What is a base case in recursion?"
            self.fail = True

        def generate_question(self, prompt, max_tokens=200, temperature=0.3, **kwargs):
            return self.q

        def generate(self, prompt, max_tokens=150, temperature=0.3, raise_on_error=False):
            if self.fail:
                raise LLMRequestError("evaluator: 429 quota exceeded")
            return (
                "Score: 0.7\n"
                "Strengths: Covers the base case.\n"
                "Weaknesses: Could mention the recursive case."
            )

    llm = FailFirstThenOkLLM()
    worker = _make_worker(EvaluatorAgent(llm_client=llm))
    env = _envelope({
        "sessionId": "sess-rt",
        "step": 1,
        "contextId": "ctx-recursion",
        "studentAnswer": "A base case stops recursion.",
    })
    msg = FakeMessage(env)
    published = {}

    async def fake_republish(message, env2, next_attempt, delay_ms, reason):
        published["next"] = next_attempt
        message.ack()

    worker._republish_for_retry = fake_republish
    await worker.on_message(msg)

    assert published.get("next") == 1  # retry scheduled, not DLQ
    assert worker.results == []  # no terminal result yet
    # Session was created, but the failed step added nothing (no double-count).
    session = worker.agent.get_session("sess-rt")
    assert session is not None
    assert session.attempts == 0
    assert session.answer_history == []


async def test_worker_replayed_same_messageid_deduped():
    # Idempotency by messageId (AI-COM-08): a genuine redelivery of the same
    # attempt is acknowledged without re-running the handler.
    from agents.evaluator.agent import EvaluatorAgent

    VALID = (
        "Score: 0.7\n"
        "Strengths: Covers the base case.\n"
        "Weaknesses: Could mention the recursive case."
    )

    class OkLLM:
        def generate_question(self, prompt, max_tokens=200, temperature=0.3, **kwargs):
            return "What is a base case in recursion?"

        def generate(self, prompt, max_tokens=150, temperature=0.3, raise_on_error=False):
            return VALID

    worker = _make_worker(EvaluatorAgent(llm_client=OkLLM()))
    env = _envelope({
        "sessionId": "sess-idem",
        "step": 1,
        "contextId": "ctx-recursion",
        "studentAnswer": "A base case stops recursion.",
    })
    msg_once = FakeMessage(env)
    msg_twice = FakeMessage(env)  # same messageId → same idempotency key

    await worker.on_message(msg_once)
    assert msg_once.acked and not msg_once.nacked
    assert len(worker.results) == 1

    worker.results = []
    await worker.on_message(msg_twice)
    assert msg_twice.acked and not msg_twice.nacked
    assert len(worker.results) == 0  # duplicate not re-processed → no result

    session = worker.agent.get_session("sess-idem")
    assert session.attempts == 1  # not double-counted


async def test_worker_persists_session_across_retries():
    # Session survives in the state store after a completed step, so a worker
    # restart / retry can rehydrate it (retry-safe session state).
    from agents.evaluator.agent import EvaluatorAgent

    VALID = (
        "Score: 0.7\n"
        "Strengths: Covers the base case.\n"
        "Weaknesses: Could mention the recursive case."
    )

    class OkLLM:
        def generate_question(self, prompt, max_tokens=200, temperature=0.3, **kwargs):
            return "What is a base case in recursion?"

        def generate(self, prompt, max_tokens=150, temperature=0.3, raise_on_error=False):
            return VALID

    worker = _make_worker(EvaluatorAgent(llm_client=OkLLM()))
    await worker.on_message(FakeMessage(_envelope({
        "sessionId": "sess-persist",
        "step": 1,
        "contextId": "ctx-recursion",
        "studentAnswer": "A base case stops recursion.",
    })))
    serialized = await worker.session_store.get("sess-persist")
    assert serialized is not None
    restored = json.loads(serialized)
    assert restored["session_id"] == "sess-persist"
    assert restored["attempts"] == 1
