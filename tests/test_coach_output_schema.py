"""COACH-05: strict coach LLM output schema and extraction.

Covers `agents.coach.models.schemas.CoachOutput` and
`agents.coach.decision.output_parser`:
- schema constraints: nudge_text 1–500, intensity 0.0–1.0, category enum
- structured extraction: nested `nudge`, top-level fields, legacy `message`
- parsing/validation failures → sanitized error in the job result, never raw
  LLM content
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agents.coach.decision.llm_decider import decide_with_llm
from agents.coach.decision.output_parser import (
    SANITIZED_LLM_PARSE_ERROR,
    SANITIZED_OUTPUT_VALIDATION_ERROR,
    CoachOutputError,
    extract_coach_output,
    parse_response,
    safe_fallback_nudge,
)
from agents.coach.models.schemas import (
    CoachAction,
    CoachInput,
    CoachOutput,
    FatigueState,
    FocusState,
    ScheduledTask,
)
from workers.coach_worker import CoachWorker

NOW = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)

VALID_NUDGE = {
    "nudge_text": "Take a 5-minute break, then continue.",
    "intensity": 0.6,
    "category": "break",
}


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


def _llm_decision(**extra) -> dict:
    return {
        "action_type": "encourage",
        "message": "Keep going!",
        "reasoning": "User is focused",
        "target_task_id": "t1",
        **extra,
    }


# ------------------------------------------------------------ schema rules #

def test_coach_output_valid():
    out = CoachOutput(**VALID_NUDGE)
    assert out.nudge_text == VALID_NUDGE["nudge_text"]
    assert out.intensity == 0.6
    assert out.category == "break"


@pytest.mark.parametrize(
    "bad",
    [
        {"nudge_text": "", "intensity": 0.5, "category": "motivation"},
        {"nudge_text": "x" * 501, "intensity": 0.5, "category": "motivation"},
        {"nudge_text": "ok", "intensity": -0.1, "category": "motivation"},
        {"nudge_text": "ok", "intensity": 1.1, "category": "motivation"},
        {"nudge_text": "ok", "intensity": 0.5, "category": "gossip"},
    ],
)
def test_coach_output_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        CoachOutput(**bad)


def test_coach_output_boundaries_accepted():
    CoachOutput(nudge_text="x" * 500, intensity=0.0, category="focus")
    CoachOutput(nudge_text="ok", intensity=1.0, category="fatigue")


# ----------------------------------------------------------- parse_response #

def test_parse_response_str_and_dict():
    assert parse_response(json.dumps(VALID_NUDGE)) == VALID_NUDGE
    assert parse_response(VALID_NUDGE) is VALID_NUDGE


def test_parse_response_fenced_json():
    raw = f"```json\n{json.dumps(VALID_NUDGE)}\n```"
    assert parse_response(raw) == VALID_NUDGE


def test_parse_response_garbage_raises_sanitized():
    with pytest.raises(CoachOutputError) as exc:
        parse_response("this is { not json at all")
    assert str(exc.value) == SANITIZED_LLM_PARSE_ERROR


def test_parse_response_non_object_json_raises():
    with pytest.raises(CoachOutputError):
        parse_response("[1, 2, 3]")


# ------------------------------------------------------------ extraction #

def test_extract_nested_nudge():
    out = extract_coach_output(_llm_decision(nudge=VALID_NUDGE))
    assert out == CoachOutput(**VALID_NUDGE)


def test_extract_top_level_fields():
    payload = _llm_decision(**VALID_NUDGE)
    assert extract_coach_output(payload) == CoachOutput(**VALID_NUDGE)


def test_extract_message_fallback_with_defaults():
    old_shape = _llm_decision()  # action_type=encourage, message only
    out = extract_coach_output(old_shape)
    assert out.nudge_text == "Keep going!"
    assert out.intensity == 0.5
    assert out.category == "motivation"


def test_extract_message_fallback_derives_category_from_action():
    payload = _llm_decision(action_type="suggest_break", message="Rest now")
    out = extract_coach_output(payload)
    assert out.category == "break"
    nudge_payload = _llm_decision(action_type="nudge", message="Focus")
    assert extract_coach_output(nudge_payload).category == "focus"


def test_extract_missing_nudge_text_raises_sanitized():
    with pytest.raises(CoachOutputError) as exc:
        extract_coach_output({"action_type": "silence", "reasoning": "r"})
    assert SANITIZED_OUTPUT_VALIDATION_ERROR in str(exc.value)
    assert "nudge_text" in str(exc.value)


def test_extract_out_of_range_intensity_raises_sanitized():
    payload = _llm_decision(nudge={"nudge_text": "ok", "intensity": 2.5, "category": "focus"})
    with pytest.raises(CoachOutputError) as exc:
        extract_coach_output(payload)
    reason = str(exc.value)
    assert SANITIZED_OUTPUT_VALIDATION_ERROR in reason
    assert "nudge" in reason or "intensity" in reason
    assert "2.5" not in reason  # no raw values leak
    assert "ok" not in reason


def test_extract_bad_category_raises_sanitized():
    payload = _llm_decision(nudge={"nudge_text": "ok", "intensity": 0.5, "category": "scream"})
    with pytest.raises(CoachOutputError):
        extract_coach_output(payload)


# ------------------------------------------------------- decide_with_llm #

@patch("agents.coach.decision.llm_decider.call_gemini")
def test_decide_sets_validated_nudge(mock_call):
    mock_call.return_value = json.dumps(_llm_decision(nudge=VALID_NUDGE))
    action = decide_with_llm(make_coach_input(), trace_id="tr-1")
    assert action.action_type == "encourage"
    assert action.nudge == CoachOutput(**VALID_NUDGE)
    assert action.message == VALID_NUDGE["nudge_text"]
    assert action.coach_error is None


@patch("agents.coach.decision.llm_decider.call_gemini")
def test_decide_invalid_nudge_sanitizes(mock_call):
    mock_call.return_value = json.dumps(
        _llm_decision(nudge={"nudge_text": "secret raw text", "intensity": 3.0, "category": "break"})
    )
    action = decide_with_llm(make_coach_input(), trace_id="tr-2")
    assert action.action_type == "encourage"  # decision still valid
    assert action.coach_error is not None
    assert action.nudge == safe_fallback_nudge()
    assert action.message is None
    assert "secret raw text" not in action.coach_error
    assert "secret raw text" not in action.nudge.nudge_text


@patch("agents.coach.decision.llm_decider.call_gemini")
def test_decide_unparseable_output_sanitizes(mock_call):
    mock_call.return_value = "bleh { not valid"
    action = decide_with_llm(make_coach_input(), trace_id="tr-3")
    assert action.action_type == "silence"
    assert action.coach_error == SANITIZED_LLM_PARSE_ERROR
    assert "bleh" not in action.reasoning
    assert action.message is None


@patch("agents.coach.decision.llm_decider.call_gemini")
def test_decide_malformed_decision_sanitizes(mock_call):
    mock_call.return_value = json.dumps(
        {"action_type": "monday", "reasoning": "weird", "message": None}
    )
    action = decide_with_llm(make_coach_input(), trace_id="tr-4")
    assert action.action_type == "silence"
    assert action.coach_error is not None
    assert SANITIZED_OUTPUT_VALIDATION_ERROR in action.coach_error


# ------------------------------------------------------- worker roundtrip #

def test_worker_payload_roundtrips_nudge_and_error():
    worker = CoachWorker()
    action = CoachAction(
        action_type="nudge",
        message="Take a break",
        reasoning="fatigue",
        nudge=CoachOutput(**VALID_NUDGE),
        coach_error=None,
    )
    payload = worker._coach_payload(action)
    assert payload["nudge"]["nudge_text"] == VALID_NUDGE["nudge_text"]
    assert payload["coach_error"] is None

    failed = CoachAction(
        action_type="silence",
        message=None,
        reasoning="Invalid coach output",
        nudge=safe_fallback_nudge(),
        coach_error=SANITIZED_LLM_PARSE_ERROR,
    )
    failed_payload = worker._coach_payload(failed)
    assert failed_payload["coach_error"] == SANITIZED_LLM_PARSE_ERROR
    assert failed_payload["nudge"]["nudge_text"] == safe_fallback_nudge().nudge_text