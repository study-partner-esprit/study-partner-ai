"""Validation / rejection pipeline for evaluator outputs (F04 / EVAL-06).

Ensures malformed, ungrounded, or incoherent LLM outputs are caught before
they reach the UI. One correction retry is attempted; if that also fails the
step is FAILED with a sanitized reason. Full LLM responses are logged for
debugging.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from agents.evaluator.prompts import validate_question as _validate_question

logger = logging.getLogger(__name__)

_REQUIRED_ANALYSIS_FIELDS = ("score", "strengths", "weaknesses")
_MIN_ANSWER_CHARS = 1
_MAX_SCORE = 1.0
_MIN_SCORE = 0.0


class AnalysisValidationError(Exception):
    """Raised when the LLM analysis output fails validation."""

    def __init__(self, reason: str, raw_response: str = ""):
        self.reason = reason
        self.raw_response = raw_response
        super().__init__(reason)


def validate_analysis_output(
    parsed: dict,
    student_answer: str,
    key_concepts: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Validate the parsed LLM analysis response before building output.

    Checks performed:
      - required fields present (score, strengths, weaknesses)
      - score is a float in [0.0, 1.0]
      - student answer is long enough to support evidence (≥ 1 char)
      - evidence can be built from the answer (answer not blank)

    Returns:
        (is_valid, sanitized_reason) — ``sanitized_reason`` is empty when valid.
    """
    if not isinstance(parsed, dict):
        return False, "LLM returned non-dict parsed output"

    # --- required fields ---------------------------------------------------
    missing = [f for f in _REQUIRED_ANALYSIS_FIELDS if f not in parsed]
    if missing:
        return False, f"missing required fields: {', '.join(missing)}"

    # --- score range -------------------------------------------------------
    score = parsed.get("score")
    if score is None:
        return False, "score could not be extracted from LLM response"
    if not isinstance(score, (int, float)):
        return False, "score is not numeric"
    if not (_MIN_SCORE <= float(score) <= _MAX_SCORE):
        return False, f"score out of range: {score}"

    # --- answer grounding --------------------------------------------------
    answer = (student_answer or "").strip()
    if len(answer) < _MIN_ANSWER_CHARS:
        return False, "student answer too short to ground evidence"

    # --- strengths / weaknesses non-empty ----------------------------------
    strengths = (parsed.get("strengths") or "").strip()
    weaknesses = (parsed.get("weaknesses") or "").strip()
    if not strengths and not weaknesses:
        return False, "LLM returned empty strengths and weaknesses"

    return True, ""


def validate_question_output(question: str) -> Tuple[bool, str]:
    """Validate a generated Socratic question for coherence.

    Returns (is_valid, sanitized_reason).
    """
    if not question or not isinstance(question, str):
        return False, "empty or non-string question"

    validated = _validate_question(question)
    if not validated:
        return False, "question failed validation (too short / no concept)"

    return True, ""


def build_failure_analysis(student_answer: str, reason: str):
    """Build a neutral fallback LLMAnalysisResponse after validation failure.

    Returns a tuple (analysis, mastery_score) so the session can continue
    with a neutral score while the rejection is logged.
    """
    from agents.evaluator.schemas import LLMAnalysisResponse

    analysis = LLMAnalysisResponse(
        concept_coverage=0.3,
        logical_coherence=0.3,
        causal_reasoning=0.3,
        error_awareness=0.3,
        answer_feedback=f"Evaluation step skipped: {reason}",
        guessing_detected=False,
        missing_concepts=[],
        misconceptions=[],
    )
    return analysis, 0.3
