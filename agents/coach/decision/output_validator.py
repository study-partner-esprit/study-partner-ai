"""Coach output validation & sanitization (F03 / COACH-06).

Gates every LLM-produced nudge before it can reach the user:

- shape: missing required fields / out-of-range intensity are rejected
- plain-text sanitization: HTML tags stripped — the frontend renders plain
  text only
- content policy: a curated term list (primary) plus an optional LLM guard
  (fallback) reject self-harm, harassment, violence and unsafe content
- after ONE correction retry the pipeline FAILS with a sanitized reason
  (see `llm_decider.decide_with_llm` and `CoachWorker`)

Mirrors `agents/planner/output_validation.py` (PLAN-04/05): a pure function
returns a list of problems; the worker escalates to TerminalError → job FAILED.
"""

from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Tuple

from agents.coach.models.schemas import CoachOutput

# (safe, sanitized_reason) — a content-guard provider used as LLM fallback.
ContentGuard = Callable[[str], Tuple[bool, str]]


class CoachOutputRejectedError(ValueError):
    """Final rejection after the one correction retry — message is sanitized."""


# ------------------------------------------------------------------ policy #

# Primary list-based content filter. Phrase-based so benign words are not
# flagged; categories follow the COACH-06 acceptance criteria.
CONTENT_POLICY_TERMS: dict = {
    "self-harm": [
        "self harm",
        "self-harm",
        "kill yourself",
        "hurt yourself",
        "cut yourself",
        "suicide",
    ],
    "harassment": [
        "you are worthless",
        "you are stupid",
        "you are a failure",
        "nobody cares about you",
        "give up on life",
        "shut up",
        "hate you",
    ],
    "violence": [
        "i will kill you",
        "shoot you",
        "stab you",
        "murder you",
        "beat you up",
    ],
    "unsafe": [
        "make a bomb",
        "bomb recipe",
        "poison recipe",
    ],
}

# Extra per-category terms via env: "self-harm:term1,harassment:term2".

_TERM_PATTERNS: List[Tuple[str, "re.Pattern"]] = [
    (
        category,
        re.compile(
            r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b",
            re.IGNORECASE,
        ),
    )
    for category, terms in CONTENT_POLICY_TERMS.items()
    if terms
]


def match_content_policy(text: str) -> Optional[str]:
    """Return the first matching policy category, or None if clean."""
    lowered = (text or "").lower()
    for category, pattern in _TERM_PATTERNS:
        if pattern.search(lowered):
            return category
    # env-configured extras (substring match is enough for these).
    extra = os.getenv("COACH_POLICY_EXTRA_TERMS", "").strip()
    for entry in extra.split(","):
        category, _, term = entry.partition(":")
        term = term.strip().lower()
        if term and category.strip() and term in lowered:
            return category.strip()
    return None


# ------------------------------------------------------------ sanitization #

_BLOCK_TAGS = re.compile(
    r"<\s*(?:script|style)\b[^>]*>.*?<\s*/\s*(?:script|style)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_ANY_TAG = re.compile(r"<[^>]*>")
_WHITESPACE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Reduce a string to plain text: drop scripts/styles and all tags."""
    if not text:
        return ""
    cleaned = _BLOCK_TAGS.sub(" ", text)
    cleaned = _ANY_TAG.sub(" ", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    return _PUNCT.sub(r"\1", cleaned)


_PUNCT = re.compile(r"\s+([.,!?;:])")


def sanitize_nudge(output: CoachOutput) -> CoachOutput:
    """Rebuild a CoachOutput whose nudge_text is guaranteed plain text."""
    return CoachOutput(
        nudge_text=strip_html(output.nudge_text),
        intensity=output.intensity,
        category=output.category,
    )


# --------------------------------------------------------------- checking #

def check_coach_output(
    output: CoachOutput,
    content_guard: Optional[ContentGuard] = None,
) -> List[str]:
    """Return a list of sanitized problems (empty list = acceptable).

    Shape checks are belt-and-braces (CoachOutput already enforces bounds);
    the content policy term list runs first, with the LLM guard as fallback.
    """
    problems: List[str] = []
    if not (1 <= len(output.nudge_text) <= 500):
        problems.append("nudge_text length out of range")
    if not (0.0 <= output.intensity <= 1.0):
        problems.append("intensity out of range")

    hit = match_content_policy(output.nudge_text)
    if hit:
        problems.append(f"content policy: {hit}")

    if not problems and content_guard is not None:
        safe, reason = content_guard(output.nudge_text)
        if not safe:
            problems.append(f"content policy: {reason or 'flagged by LLM guard'}")

    return problems


def gemini_content_guard(text: str) -> Tuple[bool, str]:
    """Default LLM-guard fallback. Fails open (list filter stays the gate)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "dummy_key_for_testing":
        return True, ""
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            "You are a strict content safety guard for a study coach. "
            "Classify the coaching message below. Reply with exactly one word: "
            "SAFE or UNSAFE. UNSAFE means it contains self-harm, harassment, "
            "violence, hate, sexual exploitation of minors, or instructions for "
            "doing harm.\n\nMessage: " + text
        )
        verdict = (response.text or "").strip().upper()
        if verdict.startswith("SAFE"):
            return True, ""
        return False, "flagged by LLM guard"
    except Exception:
        return True, ""