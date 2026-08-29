"""COACH-06: coach output validation & sanitization.

Covers `agents.coach.decision.output_validator` and the one-correction-retry
pipeline in `decide_with_llm`:
- HTML/tag stripping → plain text only
- content-policy term filter (self-harm, harassment, violence, unsafe)
- LLM-guard fallback (injected)
- one correction retry, then FAILED (CoachOutputRejectedError) with a
  sanitized reason
- silence decisions need no nudge
- CoachWorker escalates rejection → TerminalError (job FAILED)
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from agents.coach.decision.llm_decider import decide_with_llm
from agents.coach.decision.output_validator import (
    CoachOutputRejectedError,
    check_coach_output,
    gemini_content_guard,
    match_content_policy,
    sanitize_nudge,
    strip_html,
)
from agents.coach.models.schemas import (
    CoachInput,
    CoachOutput,
    FatigueState,
    FocusState,
    ScheduledTask,
)

NOW = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)


def make_coach_input() -> CoachInput:
    return CoachInput(
        scheduled_tasks=[
            ScheduledTask(
                task_id="t1",
                title="Solve exercises",
                start_time=NOW,
                end_time=NOW,
                priority=1,
            )
        ],
        current_time=NOW,
        focus_state=FocusState(state="Drifting", score=0.4),
        fatigue_state=FatigueState(state="Moderate", score=0.6),
        affective_state="engaged",
        ignored_count=0,
        do_not_disturb=False,
        is_late=False,
    )


def _decision(**nudge) -> dict:
    return {
        "action_type": "encourage",
        "message": "Keep going!",
        "reasoning": "User is focused",
        "target_task_id": "t1",
        "nudge": nudge,
    }


def _nudge(text: str, **extra) -> dict:
    return {"nudge_text": text, "intensity": 0.5, "category": "motivation", **extra}


# ------------------------------------------------------------ sanitization #

def test_strip_html_removes_tags():
    assert strip_html("<b>Focus</b> on the <i>task</i>.") == "Focus on the task."


def test_strip_html_removes_script_and_style_blocks():
    text = "Take a break <script>alert(1)</script> and hydrate."
    assert strip_html(text) == "Take a break and hydrate."
    assert strip_html("<style>body{display:none}</style>Real") == "Real"


def test_strip_html_empty_input():
    assert strip_html("") == ""
    assert strip_html("<script>x</script>") == ""


def test_sanitize_nudge_rebuilds_plain_text():
    out = CoachOutput(nudge_text="<p>Nice or <b>short</b> break</p>", intensity=0.5, category="break")
    cleaned = sanitize_nudge(out)
    assert cleaned.nudge_text == "Nice or short break"
    assert cleaned.intensity == 0.5
    assert cleaned.category == "break"


# --------------------------------------------------------- content policy #

def test_content_policy_detects_categories():
    assert match_content_policy("Take a break from self harm") == "self-harm"
    assert match_content_policy("Nobody cares about you anyway") == "harassment"
    assert match_content_policy("I will kill you if you fail") == "violence"
    assert match_content_policy("Here is how to make a bomb") == "unsafe"


def test_content_policy_ignores_benign_text():
    assert match_content_policy("Keep going, you can do it.") is None
    assert match_content_policy("") is None


def test_content_policy_extra_terms_env(monkeypatch):
    monkeypatch.setenv("COACH_POLICY_EXTRA_TERMS", "unsafe:catapult,harassment:so dumb")
    text = "your plan is so dumb"
    assert match_content_policy(text) == "harassment"
    assert match_content_policy("a good catapult for physics lab") == "unsafe"


# ------------------------------------------------------------------ checks #

def test_check_coach_output_accepts_clean():
    out = CoachOutput(**{"nudge_text": "Good progress!", "intensity": 0.4, "category": "motivation"})
    assert check_coach_output(out) == []


def test_check_coach_output_flags_policy_hit():
    out = CoachOutput(**{"nudge_text": "Just remember to end self harm", "intensity": 0.5, "category": "fatigue"})
    assert check_coach_output(out) == ["content policy: self-harm"]


def test_check_coach_output_uses_llm_guard_fallback():
    out = CoachOutput(**{"nudge_text": "innocent text", "intensity": 0.5, "category": "focus"})
    guard_false = lambda text: (False, "flagged by LLM guard")  # noqa: E731
    guard_true = lambda text: (True, "")  # noqa: E731
    assert check_coach_output(out, content_guard=guard_false) == [
        "content policy: flagged by LLM guard"
    ]
    assert check_coach_output(out, content_guard=guard_true) == []


def test_gemini_content_guard_disabled_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert gemini_content_guard("any text") == (True, "")


# ---------------------------------------------------------- decide_with_llm #

@patch("agents.coach.decision.llm_decider.call_gemini")
def test_decide_sanitizes_injected_html(mock_call):
    mock_call.return_value = json.dumps(_decision(**_nudge("Keep going <script>alert(1)</script>!")))
    action = decide_with_llm(make_coach_input(), trace_id="tr-1")
    assert action.message == "Keep going!"
    assert "alert" not in action.message
    assert "<" not in action.nudge.nudge_text


@patch("agents.coach.decision.llm_decider.call_gemini")
def test_decide_rejects_unsafe_then_accepts_corrected(mock_call):
    unsafe = json.dumps(_decision(**_nudge("Nobody cares about you, give up")))
    safe = json.dumps(_decision(**_nudge("You can do this, I believe in you")))
    mock_call.side_effect = [unsafe, safe]

    action = decide_with_llm(make_coach_input(), trace_id="tr-2")
    assert mock_call.call_count == 2  # one correction retry
    assert action.nudge.nudge_text == "You can do this, I believe in you"


@patch("agents.coach.decision.llm_decider.call_gemini")
def test_decide_fails_terminal_after_two_unsafe(mock_call):
    unsafe = json.dumps(_decision(**_nudge("I will kill you")))
    mock_call.return_value = unsafe

    with pytest.raises(CoachOutputRejectedError) as exc:
        decide_with_llm(make_coach_input(), trace_id="tr-3")
    assert mock_call.call_count == 2
    reason = str(exc.value)
    assert "coach output rejected" in reason
    assert "violence" in reason
    assert "I will kill you" not in reason  # sanitized


@patch("agents.coach.decision.llm_decider.call_gemini")
def test_decide_honors_injected_guard(mock_call):
    mock_call.return_value = json.dumps(_decision(**_nudge("totally clean words")))
    with pytest.raises(CoachOutputRejectedError):
        decide_with_llm(
            make_coach_input(),
            trace_id="tr-4",
            content_guard=lambda text: (False, "flagged by LLM guard"),
        )
    assert mock_call.call_count == 2


@patch("agents.coach.decision.llm_decider.call_gemini")
def test_decide_silence_returns_without_nudge(mock_call):
    mock_call.return_value = json.dumps(
        {"action_type": "silence", "message": None, "reasoning": "focused", "target_task_id": None}
    )
    action = decide_with_llm(make_coach_input(), trace_id="tr-5")
    assert mock_call.call_count == 1  # validation never even runs
    assert action.action_type == "silence"
    assert action.nudge is None


# ------------------------------------------------------------- worker path #

class _FakeOrchestrator:
    def __init__(self, outcome):
        self._outcome = outcome

    def run_coach(self, **kwargs):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def test_worker_fails_job_terminal_on_output_rejection():
    import asyncio

    from messaging.failures import TerminalError
    from workers.coach_worker import CoachWorker

    worker = CoachWorker(orchestrator=_FakeOrchestrator(CoachOutputRejectedError("coach output rejected: content policy: violence")))
    envelope = _Envelope()

    async def drive():
        with pytest.raises(TerminalError) as exc:
            await worker.handle({"signals": [], "messages": []}, envelope)
        return str(exc.value)

    reason = asyncio.run(drive())
    assert "coach output rejected" in reason


class _Envelope:
    userId = "user-1"
    correlationId = "corr-1"


def test_worker_returns_valid_action():
    import asyncio

    from agents.coach.models.schemas import CoachAction
    from workers.coach_worker import CoachWorker

    action = CoachAction(
        action_type="nudge",
        message="Good progress!",
        reasoning="r",
        nudge=CoachOutput(nudge_text="Good progress!", intensity=0.4, category="motivation"),
    )
    worker = CoachWorker(orchestrator=_FakeOrchestrator(action))

    async def drive():
        return await worker.handle({"signals": [], "messages": []}, _Envelope())

    result = asyncio.run(drive())
    assert result["nudge"]["nudge_text"] == "Good progress!"
    assert result["action_type"] == "nudge"