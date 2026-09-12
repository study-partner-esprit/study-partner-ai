import json
import os
from datetime import datetime

from agents.coach.models.schemas import CoachInput, CoachAction, ScheduledTask
from agents.coach.decision.prompt import SYSTEM_PROMPT, build_user_prompt
from agents.coach.decision.output_parser import (
    SANITIZED_OUTPUT_VALIDATION_ERROR,
    CoachOutputError,
    extract_coach_output,
    parse_response,
)
from agents.coach.decision.output_validator import (
    CoachOutputRejectedError,
    check_coach_output,
    gemini_content_guard,
    sanitize_nudge,
)
from agents.coach.context.preprocess import window_history
from messaging.failures import RetryableError
from pydantic import ValidationError
from utils.llm_client import LLMRequestError, MissingMockResponderError, ask
from utils.logger import get_logger

logger = get_logger(__name__)


def call_gemini(system_prompt: str, user_prompt: str, trace_id: str = "") -> str:
    # COACH-07: routed through the shared LiteLLM client (utils/llm_client.py).
    # Under LLM_MOCK=1 (or with no provider key) ask returns the mock responder;
    # that is the intended degraded behaviour, not an outage.
    try:
        return ask(
            "coach",
            system_prompt,
            user_prompt,
            trace_id=trace_id,
            mock_fn=get_mock_gemini_response,
        )
    except MissingMockResponderError:
        # Defensive: the mock_fn above means this only happens if ask's mock
        # responder itself is unavailable. Quietly degrade to the mock output.
        logger.info("coach_llm_unavailable_mock_fallback", extra={"trace_id": trace_id})
        return get_mock_gemini_response(user_prompt)
    except LLMRequestError as exc:
        # COACH-08: timeout/quota/transient infra failures are raised so the
        # job-bus retry policy owns recovery (AI-COM-06), never fabricated
        # into a fake decision. CoachWorker retries and falls back to the rule
        # engine on the final attempt.
        logger.warning("coach_llm_api_error", extra={"error": str(exc), "trace_id": trace_id})
        raise RetryableError(f"coach LLM unavailable: {exc}") from exc


def get_mock_gemini_response(user_prompt: str) -> str:
    """
    Generate intelligent mock responses based on student state for comprehensive testing.
    Analyzes the user_prompt to determine appropriate coaching response.
    """
    # Extract key information from the prompt
    focus_state = "Drifting"  # default
    fatigue_prob = 0.5  # default
    affective_state = "engaged"  # default
    is_late = False  # default

    if "Focused" in user_prompt:
        focus_state = "Focused"
    elif "Lost" in user_prompt:
        focus_state = "Lost"

    # Extract fatigue probability
    import re

    fatigue_match = re.search(r"fatigue_probability.*?(\d+\.\d+)", user_prompt)
    if fatigue_match:
        fatigue_prob = float(fatigue_match.group(1))

    # Extract affective state
    if "frustrated" in user_prompt:
        affective_state = "frustrated"
    elif "stressed" in user_prompt:
        affective_state = "stressed"
    elif "bored" in user_prompt:
        affective_state = "bored"
    elif "confident" in user_prompt:
        affective_state = "confident"

    # Extract is_late
    if "is_late.*?:.*?true" in user_prompt or "late" in user_prompt.lower():
        is_late = True
    elif "stressed" in user_prompt:
        affective_state = "stressed"
    elif "bored" in user_prompt:
        affective_state = "bored"
    elif "confident" in user_prompt:
        affective_state = "confident"

    # Generate appropriate response based on states
    if focus_state == "Lost" and fatigue_prob > 0.7:
        # High fatigue + lost focus
        if fatigue_prob > 0.9 and is_late:
            # Extremely tired and late = suspend session
            return json.dumps(
                {
                    "action_type": "suggest_break",
                    "message": "You're extremely tired and it's getting late. Let's suspend this session and continue tomorrow when you're fresh.",
                    "reasoning": "Critical fatigue levels combined with late hour indicate need for full rest.",
                    "target_task_id": None,
                    "schedule_changes": {
                        "action": "suspend_session",
                        "reasoning": "Coach detected extreme fatigue late at night and suspended session",
                    },
                }
            )
        else:
            # Regular high fatigue = suggest break with duration based on fatigue level
            break_duration = 5 if fatigue_prob <= 0.8 else 10
            return json.dumps(
                {
                    "action_type": "suggest_break",
                    "message": f"You seem quite tired. How about taking a {break_duration}-minute break to recharge?",
                    "reasoning": "High fatigue levels combined with lost focus indicate need for rest.",
                    "target_task_id": None,
                    "schedule_changes": {
                        "action": "add_break",
                        "duration_minutes": break_duration,
                        "affected_task_ids": [],
                        "reasoning": f"Coach detected high fatigue ({fatigue_prob:.1f}) and suggested {break_duration}-minute break",
                    },
                }
            )
    elif affective_state == "frustrated" and focus_state == "Lost":
        # Frustrated + lost = encourage with empathy
        return json.dumps(
            {
                "action_type": "encourage",
                "message": "I can see this is challenging. Remember, every expert was once a beginner. You've got this!",
                "reasoning": "Frustration with lost focus suggests need for motivational support.",
                "target_task_id": None,
            }
        )
    elif affective_state == "bored" and focus_state == "Drifting":
        # Bored + drifting = nudge to refocus
        return json.dumps(
            {
                "action_type": "nudge",
                "message": "Let's bring your attention back to the task. What's the next step you need to take?",
                "reasoning": "Boredom with drifting focus needs gentle redirection.",
                "target_task_id": None,
            }
        )
    elif affective_state == "confident" and focus_state == "Focused":
        # Confident + focused = positive reinforcement
        return json.dumps(
            {
                "action_type": "encourage",
                "message": "Excellent focus! You're in the zone - keep riding this momentum!",
                "reasoning": "High confidence with strong focus deserves positive reinforcement.",
                "target_task_id": None,
            }
        )
    elif fatigue_prob > 0.6:
        # General high fatigue
        if fatigue_prob > 0.9 and is_late:
            # Extremely tired and late = suspend session
            return json.dumps(
                {
                    "action_type": "suggest_break",
                    "message": "You're working very hard but seem extremely fatigued, and it's late. Let's call it a night and resume tomorrow.",
                    "reasoning": "Critical fatigue levels late at night warrant session suspension.",
                    "target_task_id": None,
                    "schedule_changes": {
                        "action": "suspend_session",
                        "reasoning": "Coach detected extreme fatigue late at night and suspended session",
                    },
                }
            )
        else:
            # Regular fatigue = suggest break with duration based on fatigue level
            break_duration = 5 if fatigue_prob <= 0.75 else 10
            return json.dumps(
                {
                    "action_type": "suggest_break",
                    "message": f"You're working hard! A {break_duration}-minute break might help you maintain quality work.",
                    "reasoning": "Elevated fatigue levels suggest rest would be beneficial.",
                    "target_task_id": None,
                    "schedule_changes": {
                        "action": "add_break",
                        "duration_minutes": break_duration,
                        "affected_task_ids": [],
                        "reasoning": f"Coach detected elevated fatigue ({fatigue_prob:.1f}) and suggested {break_duration}-minute break",
                    },
                }
            )
    elif affective_state == "stressed":
        # Stressed = calming encouragement
        return json.dumps(
            {
                "action_type": "encourage",
                "message": "Take a deep breath. You're capable and prepared for this challenge.",
                "reasoning": "Stress levels indicate need for calming, confidence-building support.",
                "target_task_id": None,
            }
        )
    else:
        # Default encouraging response
        return json.dumps(
            {
                "action_type": "encourage",
                "message": "You're doing great! Keep up the good work.",
                "reasoning": "Student shows good engagement and focus levels.",
                "target_task_id": None,
            }
        )


def decide_with_llm(
    input_data: CoachInput,
    recent_history: list | None = None,
    trace_id: str = "",
    content_guard=None,
) -> CoachAction:
    # COACH-03: never dump the full CoachInput into the prompt. Only trusted,
    # system-derived state is rendered verbatim; every user-supplied string is
    # wrapped by prompt_guard inside build_user_prompt.
    state = {
        "focus_state": input_data.focus_state.state,
        "focus_score": input_data.focus_state.score,
        "fatigue_state": input_data.fatigue_state.state,
        "fatigue_score": input_data.fatigue_state.score,
        "affective_state": input_data.affective_state,
        "ignored_count": input_data.ignored_count,
        "do_not_disturb": input_data.do_not_disturb,
        "is_late": input_data.is_late,
        "current_time": _iso(input_data.current_time),
    }
    tasks = [_task_line(t) for t in input_data.scheduled_tasks]

    task_context = None
    if any(
        [
            input_data.current_task_title,
            input_data.current_task_difficulty,
            input_data.current_task_subject,
            input_data.current_task_key_concepts,
        ]
    ):
        task_context = {
            "title": input_data.current_task_title,
            "difficulty": input_data.current_task_difficulty,
            "subject": input_data.current_task_subject,
            "key_concepts": input_data.current_task_key_concepts,
        }

    # COACH-04: only the most recent, relevant coaching history reaches the
    # prompt (windowed by count and age, configurable via env).
    recent_history = window_history(recent_history) if recent_history else recent_history

    user_prompt = build_user_prompt(
        state,
        scheduled_tasks=tasks,
        recent_history=recent_history,
        task_context=task_context,
    )

    # COACH-05 + COACH-06: LLM output is reduced to a strict CoachOutput and
    # must pass shape + content-policy validation. We probe once, then make a
    # single correction retry; on a second failure the output is FAILED with a
    # sanitized reason (CoachOutputRejectedError) — raw LLM content never
    # reaches job results.
    return _decide_with_retry(user_prompt, content_guard, trace_id)


def _decide_with_retry(user_prompt: str, content_guard, trace_id: str) -> CoachAction:
    if content_guard is None and _has_real_api_key():
        content_guard = gemini_content_guard

    problems: list[str] = []
    for attempt in range(1, 3):  # first pass + one correction retry
        prompt = user_prompt if attempt == 1 else user_prompt + _correction_note(problems)
        raw = call_gemini(SYSTEM_PROMPT, prompt, trace_id=trace_id)
        try:
            parsed = parse_response(raw)
            candidate = CoachAction(**{k: v for k, v in parsed.items() if k != "nudge"})
        except CoachOutputError as exc:
            problems = [str(exc)]
            continue
        except (ValidationError, ValueError):
            problems = [SANITIZED_OUTPUT_VALIDATION_ERROR]
            continue

        # An intentional silence needs no user-facing nudge.
        if candidate.action_type == "silence":
            candidate.message = None
            candidate.coach_error = None
            candidate.nudge = None
            _log(candidate, trace_id, coach_error=None)
            return candidate

        try:
            nudge = extract_coach_output(parsed)
            nudge = sanitize_nudge(nudge)
        except (CoachOutputError, ValidationError):
            problems = ["nudge_text length out of range"]
            continue

        problems = check_coach_output(nudge, content_guard=content_guard)
        if not problems:
            candidate.nudge = nudge
            candidate.message = nudge.nudge_text
            _log(candidate, trace_id, coach_error=None)
            return candidate

        logger.warning(
            "coach_output_rejected",
            extra={"attempt": attempt, "reason": "; ".join(problems[:3]), "trace_id": trace_id},
        )

    raise CoachOutputRejectedError(_rejected_reason(problems))


def _correction_note(problems: list[str]) -> str:
    reason = "; ".join(problems[:3]) or "validation"
    return (
        f"\n\nYour previous draft was rejected: {reason}. Reply again with ONLY "
        "a corrected JSON object. The nudge must be plain text of 1 to 500 "
        "characters and must not contain self-harm, harassment, violence or "
        "unsafe content."
    )


def _rejected_reason(problems: list[str]) -> str:
    reason = "; ".join(problems[:3])
    return f"coach output rejected: {reason}" if reason else "coach output rejected"


def _has_real_api_key() -> bool:
    key = os.getenv("GEMINI_API_KEY")
    return bool(key) and key != "dummy_key_for_testing"


def _log(action: CoachAction, trace_id: str, coach_error) -> None:
    logger.info(
        "coach_llm_action",
        extra={
            "action_type": action.action_type,
            "coach_error": coach_error,
            "trace_id": trace_id,
        },
    )


def _iso(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _task_line(task: ScheduledTask) -> dict:
    return {
        "task_id": task.task_id,
        "title": task.title,
        "start_time": task.start_time.isoformat(),
        "end_time": task.end_time.isoformat(),
        "priority": task.priority,
    }
