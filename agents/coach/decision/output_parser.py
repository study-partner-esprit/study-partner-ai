"""Coach LLM output extraction and strict validation (F03 / COACH-05).

The coach LLM must never be trusted verbatim: its response is first reduced to
a strict `CoachOutput` (nudge_text 1–500, intensity 0.0–1.0, category enum) so
the UI only ever renders validated, schema-checked text.

- `parse_response`       json.loads an LLM response, tolerating fenced blocks
- `extract_coach_output` pull + validate the nested `nudge` object
- `safe_fallback_nudge`  fixed, controlled message when extraction fails

Every failure is reported with a *sanitized* static reason — raw LLM content
never ends up in job results or reasoning strings.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from agents.coach.models.schemas import CoachOutput

SANITIZED_LLM_PARSE_ERROR = "coach output could not be parsed"
SANITIZED_OUTPUT_VALIDATION_ERROR = "coach output failed validation"

FALLBACK_NUDGE_TEXT = "Your coach does not have a new message right now."
FALLBACK_INTENSITY = 0.5
FALLBACK_CATEGORY = "motivation"

_DEFAULT_INTENSITY = 0.5
_CATEGORY_FROM_ACTION = {
    "nudge": "focus",
    "renegotiate_task": "focus",
    "suggest_break": "break",
    "encourage": "motivation",
    "silence": "motivation",
}


class CoachOutputError(ValueError):
    """Raised when the coach LLM output cannot be extracted/validated.

    `str(exc)` is always a sanitized static message (or field paths only) —
    never the raw LLM payload.
    """


def parse_response(raw) -> dict:
    """Return a dict for a raw LLM response (str or already-parsed dict).

    Tolerates triple-backtick JSON fences. Raises CoachOutputError with a
    sanitized reason on anything that cannot be parsed as an object.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise CoachOutputError(SANITIZED_LLM_PARSE_ERROR)
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        raise CoachOutputError(SANITIZED_LLM_PARSE_ERROR) from None
    if not isinstance(data, dict):
        raise CoachOutputError(SANITIZED_LLM_PARSE_ERROR)
    return data


def extract_coach_output(payload: dict) -> CoachOutput:
    """Extract and strictly validate the user-facing nudge from a parsed
    LLM response.

    - Prefers the nested `nudge` object; `nudge_text` may fall back to the
      legacy top-level `message`.
    - `intensity`/`category` default deterministically (validated whenever
      the model provides them); `nudge_text` is always required and bounded.

    Raises CoachOutputError (sanitized) on any failure.
    """
    candidate = payload.get("nudge") if isinstance(payload.get("nudge"), dict) else payload
    fields = dict(candidate or {})
    if not fields.get("nudge_text"):
        legacy = payload.get("message")
        if isinstance(legacy, str) and legacy.strip():
            fields["nudge_text"] = legacy
    if not fields.get("nudge_text"):
        raise CoachOutputError(f"{SANITIZED_OUTPUT_VALIDATION_ERROR}: nudge_text")
    if fields.get("intensity") is None:
        fields["intensity"] = _DEFAULT_INTENSITY
    if not fields.get("category"):
        fields["category"] = _default_category(payload)
    try:
        return CoachOutput(**fields)
    except ValidationError as exc:
        raise CoachOutputError(_sanitized_reason(exc)) from None


def _default_category(payload: dict) -> str:
    return _CATEGORY_FROM_ACTION.get(str(payload.get("action_type")), FALLBACK_CATEGORY)


def safe_fallback_nudge() -> CoachOutput:
    """A fixed, controlled nudge used when extraction/validation fails."""
    return CoachOutput(
        nudge_text=FALLBACK_NUDGE_TEXT,
        intensity=FALLBACK_INTENSITY,
        category=FALLBACK_CATEGORY,
    )


def _sanitized_reason(exc: ValidationError) -> str:
    """Reduce a ValidationError to field paths only — no offending values."""
    locs = sorted({"/".join(str(part) for part in e["loc"]) for e in exc.errors()})
    if not locs:
        return SANITIZED_OUTPUT_VALIDATION_ERROR
    return f"{SANITIZED_OUTPUT_VALIDATION_ERROR}: {', '.join(locs)}"