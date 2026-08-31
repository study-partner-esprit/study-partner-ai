"""CoachWorker unit tests (F03 / COACH-01) — no broker, no heavy orchestrator.

The AIOrchestrator is faked; the tests pin the worker's contract:
payload→context validation, identity from the envelope only, off-loop
execution, JSON-safe CoachAction results and the BaseAIWorker ACK/NACK
pipeline decisions.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agents.coach.models.schemas import CoachAction, ScheduleChange
from messaging.failures import RetryableError, TerminalError
from messaging.topology import MAX_RETRIES, RETRY_HEADER
from workers.coach_worker import CoachWorker
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
        "type": "study.coach.nudge",
        "version": "1",
        "userId": "user-42",
        "requestId": "req-1",
        "timestamp": "2026-08-26T08:00:00Z",
        "payload": payload,
    }


def default_action() -> CoachAction:
    return CoachAction(
        action_type="nudge",
        message="Keep going, you're on track.",
        reasoning="test default",
        target_task_id=None,
    )


class FakeCoachOrchestrator:
    """Stands in for AIOrchestrator; records run_coach kwargs, returns actions."""

    def __init__(self, behaviour=None):
        self.calls = []
        self.behaviour = behaviour or (lambda **kw: default_action())

    def run_coach(
        self,
        user_id=None,
        current_time=None,
        ignored_count=0,
        do_not_disturb=False,
        trace_id=None,
        live_focus_score=None,
        live_focus_state=None,
        live_fatigue_score=None,
        live_fatigue_state=None,
        session_stats=None,
    ):
        self.calls.append(
            {
                "user_id": user_id,
                "current_time": current_time,
                "ignored_count": ignored_count,
                "do_not_disturb": do_not_disturb,
                "trace_id": trace_id,
                "live_focus_score": live_focus_score,
                "live_focus_state": live_focus_state,
                "live_fatigue_score": live_fatigue_score,
                "live_fatigue_state": live_fatigue_state,
                "session_stats": session_stats,
            }
        )
        return self.behaviour(**self.calls[-1])


def make_worker(behaviour=None):
    orchestrator = FakeCoachOrchestrator(behaviour)

    class RecordingWorker(CoachWorker):
        def __init__(self):
            super().__init__(
                idempotency_store=InMemoryIdempotencyStore(), orchestrator=orchestrator
            )
            self.results = []

        async def _publish_result(self, env, *, status, payload=None, error=None):
            self.results.append((status, payload, error))

    return RecordingWorker(), orchestrator


async def consume(worker, message):
    await worker.on_message(message)


# ---------------------------------------------------------------- happy path

async def test_valid_payload_publishes_completed_result():
    worker, orchestrator = make_worker()
    msg = FakeMessage(
        envelope(
            {
                "ignored_count": 2,
                "do_not_disturb": False,
                "focus_score": 0.4,
                "focus_state": "Drifting",
                "fatigue_score": 0.2,
            }
        )
    )

    await consume(worker, msg)

    assert msg.acked and not msg.nacked
    called = orchestrator.calls[0]
    assert called["user_id"] == "user-42"
    assert called["ignored_count"] == 2
    assert called["do_not_disturb"] is False
    assert called["live_focus_score"] == 0.4
    assert called["live_focus_state"] == "Drifting"
    assert called["live_fatigue_score"] == 0.2
    assert called["trace_id"] == "0f8e2d1a-3b4c-4d6e-8f80-91a2b3c4d5e6"

    status, payload, error = worker.results[0]
    assert status == "completed"
    assert error is None
    assert payload["action_type"] == "nudge"
    assert payload["fallbackUsed"] is False


async def test_user_id_in_payload_is_terminal():
    # userId comes from the authenticated envelope only (COACH-02 AC);
    # CoachRequest has extra="forbid", so a body user_id is rejected.
    worker, orchestrator = make_worker()
    with pytest.raises(TerminalError):
        await consume(
            worker,
            FakeMessage(envelope({"user_id": "evil-peer", "ignored_count": 1})),
        )
    assert orchestrator.calls == []


async def test_defaults_fill_missing_optional_fields():
    worker, orchestrator = make_worker()
    await consume(worker, FakeMessage(envelope({})))

    called = orchestrator.calls[0]
    assert called["ignored_count"] == 0
    assert called["do_not_disturb"] is False
    assert called["live_focus_score"] is None
    assert called["live_focus_state"] is None
    assert called["live_fatigue_score"] is None
    assert called["live_fatigue_state"] is None
    assert called["session_stats"] is None


async def test_current_time_parsed_from_iso_string():
    worker, orchestrator = make_worker()
    await consume(
        worker, FakeMessage(envelope({"current_time": "2026-08-26T08:00:00Z"}))
    )

    parsed = orchestrator.calls[0]["current_time"]
    assert isinstance(parsed, datetime)
    assert parsed.isoformat().startswith("2026-08-26T08:00:00")


async def test_orchestrator_runs_off_event_loop_via_to_thread():
    worker, orchestrator = make_worker()
    seen_threads = []

    real_run_coach = orchestrator.run_coach

    def run_coach_records_thread(**kw):
        seen_threads.append(threading.current_thread())
        return real_run_coach(**kw)

    orchestrator.run_coach = run_coach_records_thread

    await consume(worker, FakeMessage(envelope({})))
    assert seen_threads[0] is not threading.main_thread()


async def test_lazy_orchestrator_not_built_until_first_handle():
    built = {"count": 0}

    class CountingWorker(CoachWorker):
        @property
        def orchestrator(self):
            if self._orchestrator is None:
                built["count"] += 1
                self._orchestrator = object()  # would explode if used as real one
            return self._orchestrator

    w = CountingWorker(idempotency_store=InMemoryIdempotencyStore())
    assert built["count"] == 0  # constructor did not touch the orchestrator


# ----------------------------------------------------------- invalid input

async def test_non_object_payload_is_terminal():
    worker, orchestrator = make_worker()
    msg = FakeMessage(envelope("just a string"))

    with pytest.raises(TerminalError):
        await consume(worker, msg)

    assert msg.nacked and not msg.acked
    assert orchestrator.calls == []
    # payload must be a dict per the AI-COM-02 contract → dead-lettered at
    # envelope validation, before dispatch; no result event (no attempt ran)
    assert worker.results == []


async def test_non_integer_ignored_count_is_terminal():
    worker, _ = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"ignored_count": "lots"})))


async def test_non_boolean_dnd_is_terminal():
    worker, _ = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"do_not_disturb": "yes"})))


async def test_out_of_range_score_is_terminal():
    worker, _ = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"focus_score": 1.5})))


async def test_unknown_focus_state_is_terminal():
    worker, _ = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"focus_state": "zombie"})))


async def test_invalid_current_time_is_terminal():
    worker, _ = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"current_time": "not-a-date"})))


# ------------------------------------------------- COACH-02 bounded schema

def focused_signal() -> dict:
    return {
        "timestamp": "2026-08-26T08:00:00Z",
        "focus_state": "Focused",
        "focus_score": 0.8,
        "fatigue_state": "Alert",
        "fatigue_score": 0.1,
        "focus_confidence": 0.9,
        "focus_trend": -0.05,
    }


async def test_full_session_context_accepted():
    worker, orchestrator = make_worker()
    payload = {
        "session_id": "sess-123",
        "signals": [focused_signal() for _ in range(3)],
        "messages": [
            {"role": "user", "content": "I feel drained."},
            {"role": "assistant", "content": "Take a rest."},
        ],
        "focus_state": "Drifting",
        "focus_score": 0.4,
        "fatigue_score": 0.7,
        "ignored_count": 2,
        "do_not_disturb": False,
        "current_time": "2026-08-26T08:05:00Z",
    }
    await consume(worker, FakeMessage(envelope(payload)))

    assert worker.results[0][0] == "completed"
    called = orchestrator.calls[0]
    assert called["live_focus_state"] == "Drifting"
    assert called["live_focus_score"] == 0.4
    assert called["live_fatigue_score"] == 0.7


async def test_signal_window_boundary_is_enforced():
    worker, _ = make_worker()
    ok = {"signals": [focused_signal() for _ in range(20)]}
    too_many = {"signals": [focused_signal() for _ in range(21)]}
    await consume(worker, FakeMessage(envelope(ok)))
    assert worker.results[0][0] == "completed"
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope(too_many)))


async def test_signal_extra_fields_are_terminal():
    worker, _ = make_worker()
    bad = {"signals": [{**focused_signal(), "mystery": 1}]}
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope(bad)))


async def test_message_content_length_capped_at_2000_chars():
    worker, _ = make_worker()
    long_msg = {"messages": [{"role": "user", "content": "x" * 2001}]}
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope(long_msg)))


async def test_total_payload_size_capped_at_16kb():
    worker, _ = make_worker()
    big = {
        "messages": [
            {"role": "user", "content": "y" * 2000} for _ in range(10)
        ]  # ~20 KB of content alone
    }
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope(big)))


async def test_unknown_body_fields_are_terminal():
    worker, _ = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"mystery_field": 1})))


# ------------------------------------------------------- COACH-13 session stats

def session_stats_payload():
    return {
        "progress_pct": 42,
        "minutes_elapsed": 25,
        "task_switches": 3,
        "break_count": 2,
        "current_streak_days": 7,
    }


async def test_session_stats_passed_through_to_orchestrator():
    worker, orchestrator = make_worker()
    await consume(
        worker,
        FakeMessage(envelope({"session_stats": session_stats_payload()})),
    )

    assert worker.results[0][0] == "completed"
    assert orchestrator.calls[0]["session_stats"] == session_stats_payload()


async def test_missing_session_stats_defaults_to_none():
    worker, orchestrator = make_worker()
    await consume(worker, FakeMessage(envelope({})))

    assert worker.results[0][0] == "completed"
    assert orchestrator.calls[0]["session_stats"] is None


async def test_out_of_range_session_stat_is_terminal():
    worker, _ = make_worker()
    bad = {**session_stats_payload(), "minutes_elapsed": 601}
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"session_stats": bad})))


async def test_boolean_session_stat_is_terminal():
    worker, _ = make_worker()
    bad = {**session_stats_payload(), "break_count": True}
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"session_stats": bad})))


async def test_session_stats_unknown_field_is_terminal():
    worker, _ = make_worker()
    bad = {**session_stats_payload(), "hacked": 1}
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"session_stats": bad})))


async def test_boolean_for_numeric_field_is_terminal():
    worker, _ = make_worker()
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"focus_score": True})))
    with pytest.raises(TerminalError):
        await consume(worker, FakeMessage(envelope({"ignored_count": True})))


# ------------------------------------------------------------ retry policy

async def test_transient_orchestrator_failure_schedules_retry():
    def boom(**kw):
        raise RetryableError("LLM timeout")

    worker, _ = make_worker(boom)
    msg = FakeMessage(envelope({}), headers={"x-retry-count": 0})

    published = {}

    async def fake_republish(message, env, next_attempt, delay_ms, reason):
        published["next"] = next_attempt
        message.ack()

    worker._republish_for_retry = fake_republish
    await consume(worker, msg)

    assert published["next"] == 1
    assert msg.acked
    assert worker.results == []  # no terminal result while retries remain


# ------------------------------------------------------------ JSON-safety

async def test_result_payload_is_json_safe():
    def action_with_schedule(**kw):
        return CoachAction(
            action_type="suggest_break",
            message="Take a short break.",
            reasoning="fatigue detected",
            schedule_changes=ScheduleChange(
                action="reschedule_task",
                new_start_time=datetime(2026, 8, 27, 9, 30, tzinfo=timezone.utc),
                affected_task_ids=["task-1"],
                reasoning="too fatigued to continue now",
            ),
        )

    worker, _ = make_worker(action_with_schedule)
    await consume(worker, FakeMessage(envelope({})))

    status, payload, error = worker.results[0]
    assert status == "completed"
    assert error is None
    # JSON-safe: datetimes serialized to strings, round-trip clean
    new_start = payload["schedule_changes"]["new_start_time"]
    assert isinstance(new_start, str)
    assert json.loads(json.dumps(payload)) == payload


# ------------------------------------------------------- COACH-08 fallback

async def test_retryable_below_max_attempts_propagates():
    def boom(**kw):
        raise RetryableError("coach LLM unavailable: upstream timeout")

    worker, _ = make_worker(boom)
    worker.current_attempt = 0  # underneath MAX_RETRIES → let the policy retry

    env = SimpleNamespace(userId="user-42", correlationId="tr-1")
    with pytest.raises(RetryableError):
        await worker.handle({}, env)


async def test_final_attempt_falls_back_to_rule_engine():
    from agents.coach.decision.output_validator import check_coach_output
    from agents.coach.models.schemas import CoachOutput

    def boom(**kw):
        raise RetryableError("coach LLM unavailable: quota exceeded")

    worker, orchestrator = make_worker(boom)
    msg = FakeMessage(envelope({}), headers={RETRY_HEADER: MAX_RETRIES})

    await consume(worker, msg)

    assert msg.acked and not msg.nacked
    status, payload, error = worker.results[0]
    assert status == "completed"
    assert error is None
    assert payload["fallbackUsed"] is True
    # no hard rule matched (default state) → guaranteed safe fallback nudge
    assert payload["action_type"] == "nudge"
    assert payload["coach_error"]
    # AC#3 — the fallback decision passes COACH-05/06 validation.
    assert payload["nudge"] is not None
    assert check_coach_output(CoachOutput(**payload["nudge"])) == []


async def test_final_attempt_dnd_fires_rule_engine_silence():
    def boom(**kw):
        raise RetryableError("coach LLM unavailable: connection reset")

    worker, _ = make_worker(boom)
    msg = FakeMessage(
        envelope({"do_not_disturb": True}), headers={RETRY_HEADER: MAX_RETRIES}
    )

    await consume(worker, msg)

    assert msg.acked and not msg.nacked
    status, payload, error = worker.results[0]
    assert status == "completed"
    assert payload["fallbackUsed"] is True
    assert payload["action_type"] == "silence"
    assert payload["coach_error"]


# ------------------------------------------------------- COACH-09 persistence

class FakeCoachHistoryRepo:
    def __init__(self):
        self.calls = []

    def save_action(self, *args, **kwargs):
        self.calls.append((args, kwargs))


async def test_job_completion_persists_nudge_by_correlation_id():
    worker, _ = make_worker()
    fake = FakeCoachHistoryRepo()
    worker._history_repo = fake

    await consume(worker, FakeMessage(envelope({})))

    assert worker.results[0][0] == "completed"
    assert len(fake.calls) == 1
    args, kwargs = fake.calls[0]
    user_id, action, _input, = args[:3]
    assert user_id == "user-42"
    assert action.action_type == "nudge"
    assert kwargs["trace_id"] == "0f8e2d1a-3b4c-4d6e-8f80-91a2b3c4d5e6"
    assert kwargs["correlation_id"] == "0f8e2d1a-3b4c-4d6e-8f80-91a2b3c4d5e6"


async def test_fallback_path_persists_nudge_to_history():
    def boom(**kw):
        raise RetryableError("coach LLM unavailable: quota exceeded")

    worker, _ = make_worker(boom)
    fake = FakeCoachHistoryRepo()
    worker._history_repo = fake
    msg = FakeMessage(envelope({}), headers={RETRY_HEADER: MAX_RETRIES})

    await consume(worker, msg)

    assert worker.results[0][1]["fallbackUsed"] is True
    assert len(fake.calls) == 1
    user_id, action, _input = fake.calls[0][0][:3]
    assert user_id == "user-42"
    assert action.action_type == "nudge"
    assert fake.calls[0][1]["trace_id"] == "0f8e2d1a-3b4c-4d6e-8f80-91a2b3c4d5e6"


async def test_failure_path_persists_nothing():
    def boom(**kw):
        raise RetryableError("coach LLM unavailable: upstream timeout")

    worker, _ = make_worker(boom)
    fake = FakeCoachHistoryRepo()
    worker._history_repo = fake
    worker.current_attempt = 0  # underneath MAX_RETRIES → will re-raise

    env = SimpleNamespace(userId="user-42", correlationId="tr-1")
    with pytest.raises(RetryableError):
        await worker.handle({}, env)

    assert fake.calls == []