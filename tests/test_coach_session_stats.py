"""Coach session-stats regression tests (F03 / COACH-13).

Covers the bounded SessionStats feed end-to-end on the AI side:

- `SessionStats`/`CoachSessionStats` bounds are enforced identically on both
  the agent model and the bus schema (parity with `payloadSchemas.js`)
- missing or stale stats default — they never fail the job, the prompt build,
  or the decision path
- stats reach the TRUSTED state block verbatim (they are system-derived, so
  they are NOT untrusted-blocks), and the fallback/history path receives them
  too
- the 16 KB CoachRequest cap still holds with stats attached

Covers COACH-13 AC:
- SessionStats schema with the exact bounded fields
- CoachInput extended; missing/stale stats default, never fail the job
- stats bounded so the payload stays within the 16 KB cap
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agents.coach.decision.llm_decider import decide_with_llm
from agents.coach.decision.prompt import _DECISION_INSTRUCTIONS, build_user_prompt
from agents.coach.models.schemas import (
    CoachInput,
    FatigueState,
    FocusState,
    SessionStats,
)
from workers.schemas import COACH_MAX_PAYLOAD_BYTES, CoachRequest, CoachSessionStats

VALID_STATS = {
    "progress_pct": 42,
    "minutes_elapsed": 25,
    "task_switches": 3,
    "break_count": 2,
    "current_streak_days": 7,
}


def _coach_input(session_stats=None) -> CoachInput:
    return CoachInput(
        scheduled_tasks=[],
        current_time=datetime(2026, 8, 31, 9, 0, tzinfo=timezone.utc),
        focus_state=FocusState(state="Focused", score=0.8),
        fatigue_state=FatigueState(state="Alert", score=0.2),
        affective_state="confident",
        session_stats=session_stats,
    )


def _trusted_state(prompt: str) -> dict:
    m = re.search(
        r"Student state \(TRUSTED system-derived data\):\n(\{.*?\})\n\n",
        prompt,
        re.S,
    )
    assert m, "trusted state block missing from prompt"
    return json.loads(m.group(1))


# ----------------------------------------------------------------- bounds


@pytest.mark.parametrize(
    "field,lo,hi",
    [
        ("progress_pct", 0, 100),
        ("minutes_elapsed", 0, 600),
        ("task_switches", 0, 50),
        ("break_count", 0, 20),
        ("current_streak_days", 0, 365),
    ],
)
def test_session_stats_bounds_are_enforced_in_worker_schema(field, lo, hi):
    ok = {**VALID_STATS, field: hi}
    assert CoachSessionStats(**ok)
    too_far = hi + 1
    with pytest.raises(ValidationError):
        CoachSessionStats(**{**VALID_STATS, field: too_far})
    with pytest.raises(ValidationError):
        CoachSessionStats(**{**VALID_STATS, field: lo - 1})


@pytest.mark.parametrize(
    "field,lo,hi",
    [
        ("progress_pct", 0, 100),
        ("minutes_elapsed", 0, 600),
        ("task_switches", 0, 50),
        ("break_count", 0, 20),
        ("current_streak_days", 0, 365),
    ],
)
def test_session_stats_bounds_are_enforced_in_agent_model(field, lo, hi):
    with pytest.raises(ValidationError):
        SessionStats(**{**VALID_STATS, field: hi + 1})
    with pytest.raises(ValidationError):
        SessionStats(**{**VALID_STATS, field: lo - 1})


def test_session_stats_default_all_fields_to_zero():
    assert SessionStats().model_dump() == {
        "progress_pct": 0,
        "minutes_elapsed": 0,
        "task_switches": 0,
        "break_count": 0,
        "current_streak_days": 0,
    }
    assert CoachSessionStats().model_dump() == {
        "progress_pct": 0,
        "minutes_elapsed": 0,
        "task_switches": 0,
        "break_count": 0,
        "current_streak_days": 0,
    }


def test_session_stats_rejects_booleans():
    for field in VALID_STATS:
        with pytest.raises(ValidationError):
            CoachSessionStats(**{**VALID_STATS, field: True})


def test_session_stats_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        CoachSessionStats(**{**VALID_STATS, "hacked": 1})


# ------------------------------------------------------- CoachRequest plumbing


def test_coach_request_accepts_full_session_stats():
    req = CoachRequest(session_stats=VALID_STATS)
    assert req.session_stats.progress_pct == 42
    assert req.session_stats.current_streak_days == 7


def test_coach_request_without_stats_never_fails():
    req = CoachRequest()
    assert req.session_stats is None
    assert req.to_coach_context()["session_stats"] is None


def test_to_coach_context_passes_bounded_stats_dict():
    req = CoachRequest(session_stats=VALID_STATS)
    ctx = req.to_coach_context()
    assert ctx["session_stats"] == VALID_STATS


def test_session_stats_stays_within_16kb_cap():
    # COACH-13 AC#4 — bounded stats trivially fit the existing 16 KB cap.
    size = len(
        json.dumps({"session_stats": VALID_STATS}, separators=(",", ":")).encode("utf-8")
    )
    assert size < COACH_MAX_PAYLOAD_BYTES
    CoachRequest(session_stats=VALID_STATS)  # passes the payload-size validator


# ------------------------------------------------------- prompt trusted state


def test_stats_reach_the_trusted_state_block_verbatim():
    state = _trusted_state(build_user_prompt({"session_stats": VALID_STATS}))
    assert state["session_stats"] == VALID_STATS


def test_missing_stats_render_safe_placeholder_in_state():
    state = _trusted_state(build_user_prompt({"session_stats": "not provided (defaults apply)"}))
    assert state["session_stats"] == "not provided (defaults apply)"


def test_decision_instructions_reference_session_stats():
    assert "session stats" in _DECISION_INSTRUCTIONS.lower()
    assert "progress_pct" in _DECISION_INSTRUCTIONS


# ------------------------------------------------------------- decision path


def _responder_from_state(monkeypatch):
    """Mocked LLM: derive category from the trusted state only and echo it."""
    import agents.coach.decision.llm_decider as decider

    def fake_ask(service, system_prompt, user_prompt, trace_id="", mock_fn=None):
        state = _trusted_state(user_prompt)
        focus = state["focus_state"]
        return json.dumps(
            {
                "action_type": "nudge",
                "message": "Stay on track.",
                "reasoning": "from trusted state",
                "target_task_id": None,
                "nudge": {
                    "nudge_text": "Keep it steady.",
                    "intensity": 0.8,
                    "category": "focus" if focus == "Lost" else "motivation",
                },
            }
        )

    monkeypatch.setattr(decider, "ask", fake_ask)


def test_decide_with_llm_receives_stats_and_keeps_decision_stable(monkeypatch):
    _responder_from_state(monkeypatch)
    action = decide_with_llm(_coach_input(session_stats=SessionStats(**VALID_STATS)))

    assert action.action_type == "nudge"
    assert action.nudge.category == "motivation"
    assert action.nudge.intensity == 0.8


def test_decide_with_llm_with_missing_stats_defaults_safely(monkeypatch):
    _responder_from_state(monkeypatch)
    action = decide_with_llm(_coach_input(session_stats=None))

    assert action.action_type == "nudge"
    assert action.nudge.nudge_text == "Keep it steady."


def test_decide_with_llm_partial_stats_use_bounded_defaults(monkeypatch):
    _responder_from_state(monkeypatch)
    # Only one stat supplied — the rest must default to 0, never raise.
    partial = SessionStats(progress_pct=12)
    action = decide_with_llm(_coach_input(session_stats=partial))

    assert action.action_type == "nudge"


# ------------------------------------------------------------ orchestrator


def test_orchestrator_malformed_stats_default_to_none_never_fail():
    from services.ai_orchestrator.orchestrator import AIOrchestrator

    orch = AIOrchestrator()
    # Not routed through the bus schema, an out-of-bounds dict must degrade.
    assert orch._resolve_session_stats({"minutes_elapsed": 99999}, trace_id="t") is None
    assert orch._resolve_session_stats(None, trace_id="t") is None
    assert (
        orch._resolve_session_stats(VALID_STATS, trace_id="t").model_dump()
        == VALID_STATS
    )


def test_coach_input_accepts_session_stats_field():
    inp = _coach_input(session_stats=SessionStats(**VALID_STATS))
    assert inp.session_stats.task_switches == 3
    assert _coach_input(session_stats=None).session_stats is None