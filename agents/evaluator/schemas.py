"""
Pydantic models for evaluation data structures and validation.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime, timezone
import re


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================================
# TASK EVALUATION CONTEXT
# ============================================================================

class TaskEvaluationContext(BaseModel):
    """Context for task evaluation - built in-memory from task input."""

    task_title: str = Field(..., description="Title of the task")
    task_description: str = Field(..., description="Description of the task")
    task_details: str = Field(..., description="Detailed explanation of the task")
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts to evaluate")


# ============================================================================
# ANSWER ANALYSIS RESPONSE
# ============================================================================

class LLMAnalysisResponse(BaseModel):
    """LLM response for answer analysis."""

    concept_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0-1 score for concept coverage"
    )
    logical_coherence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0-1 score for logical coherence"
    )
    causal_reasoning: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0-1 score for causal reasoning"
    )
    error_awareness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0-1 score for error awareness"
    )
    answer_feedback: str = Field(
        ...,
        description="Constructive feedback for student (1 sentence)"
    )
    guessing_detected: bool = Field(
        ...,
        description="Whether guessing/hallucination detected"
    )
    missing_concepts: List[str] = Field(
        default_factory=list,
        description="Concepts not demonstrated in answer"
    )
    misconceptions: List[str] = Field(
        default_factory=list,
        description="Incorrect ideas expressed in answer"
    )


# ============================================================================
# EVALUATION STATE
# ============================================================================

class EvaluationState(str, Enum):
    """States in the evaluation state machine."""

    MASTERY_CONFIRMED = "mastery_confirmed"
    FAILED = "failed"
    CONTINUE = "continue"


# ============================================================================
# SESSION MODEL
# ============================================================================

class SessionState(str, Enum):
    """States for interactive evaluation session."""

    ASKING = "asking"
    ANALYZING = "analyzing"
    COMPLETE = "complete"


class EvaluationSession(BaseModel):
    """Represents an interactive evaluation session - held in-memory."""

    session_id: str
    task_title: str
    task_description: str
    task_details: str
    context: TaskEvaluationContext
    state: SessionState = SessionState.ASKING
    depth_level: str = "what"
    question_history: List[str] = Field(default_factory=list)
    answer_history: List[str] = Field(default_factory=list)
    analysis_history: List[LLMAnalysisResponse] = Field(default_factory=list)
    mastery_score: float = 0.0
    attempts: int = 0
    max_attempts: int = 5
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    # EVAL-02b: target Bloom level resolved server-side from the learning
    # objective and carried in the job payload as evaluation context.  Echoed
    # in every EvaluationOutput so the backend can persist it per step.
    target_bloom_level: Optional[str] = Field(default=None)


# ============================================================================
# EVALUATION OUTPUT (EVAL-04)
# ============================================================================

_DIM_BOUNDS = dict(ge=0.0, le=1.0)


class EvidenceItem(BaseModel):
    """A single grounded evidence item (EVAL-05).

    ``quote`` is drawn verbatim from the student's answer and is bounded to
    [1, 200] characters — an empty quote means the dimension is unsupported
    and is rejected at validation. Shaped as ``{dimension, quote}`` so the
    items can be reused directly as ``competency.evidence[]`` entries in the
    F14 profile store (BLOOM-08).
    """

    model_config = ConfigDict(extra="forbid")

    dimension: str = Field(..., description="Which dimension this evidence supports")
    quote: str = Field(
        ..., min_length=1, max_length=200,
        description="Verbatim evidence quote from the answer (max 200 chars)",
    )


class EvaluationOutput(BaseModel):
    """Strict, validated schema for a single evaluation step result (EVAL-04).

    Five dimension scores are each bounded to [0.0, 1.0]; any value outside
    that range is rejected at construction. Bloom-level fields target the
    F14 learning-objective feature; until that is wired (EVAL-02b) they stay
    null unless supplied.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Evaluation session identifier")

    concept_coverage: float = Field(..., **_DIM_BOUNDS, description="0-1 concept coverage")
    logical_coherence: float = Field(..., **_DIM_BOUNDS, description="0-1 logical coherence")
    causal_reasoning: float = Field(..., **_DIM_BOUNDS, description="0-1 causal reasoning")
    error_awareness: float = Field(..., **_DIM_BOUNDS, description="0-1 error awareness")
    specificity: float = Field(..., **_DIM_BOUNDS, description="0-1 answer specificity")

    mastery_score: float = Field(..., **_DIM_BOUNDS, description="0-1 overall mastery")
    next_question: Optional[str] = Field(
        None, description="Next Socratic question when the session continues"
    )
    session_status: str = Field(
        ..., description="MASTERY_CONFIRMED, FAILED, or CONTINUE"
    )
    guessing_detected: bool = Field(
        ..., description="Whether guessing/hallucination was detected in the answer"
    )

    # EVAL-05: evidence grounding — one {dimension, quote} item per dimension,
    # drawn verbatim from the answer. A result with no evidence (un-grounded
    # scores) is rejected at validation.
    evidence: List["EvidenceItem"] = Field(
        ..., min_length=1, description="Grounded evidence quotes per dimension"
    )

    # F14 / BLOOM fields — target level (from the objective) and the
    # demonstrated level (the cognitive operation the answer actually shows);
    # the two MAY differ. Null until objective targeting is wired (EVAL-02b).
    target_bloom_level: Optional[str] = Field(
        None, description="Bloom level echoed from the learning objective"
    )
    demonstrated_bloom_level: Optional[str] = Field(
        None, description="Bloom level the answer actually demonstrates"
    )


def _specificity_score(student_answer: str, key_concepts: List[str]) -> float:
    """Deterministic specificity estimate in [0.0, 1.0].

    A specific answer is substantive (not a one-word reply) and uses the
    task's key concepts rather than vague filler.
    """
    answer = (student_answer or "").strip()
    words = answer.split()
    if not words:
        return 0.0
    length_score = min(1.0, len(words) / 20.0)          # wordiness component
    concepts = [c for c in (key_concepts or []) if len(c) >= 5]
    if not concepts:
        specificity = length_score
    else:
        answer_lower = answer.lower()
        covered = sum(1 for c in concepts if c.lower() in answer_lower)
        concept_score = covered / len(concepts)
        specificity = 0.6 * concept_score + 0.4 * length_score
    return max(0.0, min(1.0, round(specificity, 3)))


_EVIDENCE_DIMENSIONS = (
    "concept_coverage",
    "logical_coherence",
    "causal_reasoning",
    "error_awareness",
    "specificity",
)

_CAUSAL_MARKERS = re.compile(
    r"\b(because|since|therefore|however|thus|so that|leads? to|results? in|"
    r"causes?|how|why|due to|consequently|if .* then)\b",
    re.IGNORECASE,
)
_MAX_EVIDENCE_QUOTE = 200


def _clip_quote(text: str, limit: int = _MAX_EVIDENCE_QUOTE) -> str:
    """Clip ``text`` to ``limit`` chars, preferring a whole-sentence boundary."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    window = text[:limit]
    # back off to the last sentence/word boundary within the window
    boundary = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if boundary > limit // 2:
        return window[: boundary + 1].rstrip()
    boundary = window.rfind(" ")
    return window[:boundary].rstrip() if boundary > 0 else window.rstrip()


def _extract_dimension_quotes(answer: str, key_concepts: List[str]) -> dict:
    """Select one raw quote per dimension from the student's answer.

    Deterministic, verbatim-from-the-answer selection:
      - concept_coverage   -> sentence containing the most key concepts
      - causal_reasoning   -> sentence with a causal marker
      - specificity        -> the longest sentence
      - logical_coherence  -> the first sentence
      - error_awareness    -> the first sentence (awareness is inferred)
    An empty answer yields an empty mapping (→ un-grounded, rejected).
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (answer or "").strip()) if s.strip()]
    if not sentences:
        return {}

    concepts_lower = [c.lower() for c in (key_concepts or []) if len(c) >= 5]

    def concept_hits(sentence: str) -> int:
        s_lower = sentence.lower()
        return sum(1 for c in concepts_lower if c in s_lower)

    causal_sentences = [s for s in sentences if _CAUSAL_MARKERS.search(s)]

    quotes = {}
    quotes["concept_coverage"] = max(sentences, key=concept_hits)
    quotes["specificity"] = max(sentences, key=len)
    quotes["logical_coherence"] = sentences[0]
    quotes["error_awareness"] = sentences[0]
    quotes["causal_reasoning"] = causal_sentences[0] if causal_sentences else sentences[0]
    return quotes


def build_evidence(answer: str, key_concepts: Optional[List[str]] = None) -> List["EvidenceItem"]:
    """Build one ``{dimension, quote}`` evidence item per dimension.

    Quotes are clipped to a max of 200 characters and validated (empty quotes
    are rejected). If the answer is empty/unsupported, no evidence is produced
    so an ``EvaluationOutput`` built from it fails validation.
    """
    quotes = _extract_dimension_quotes(answer, key_concepts)
    items: List[EvidenceItem] = []
    for dimension in _EVIDENCE_DIMENSIONS:
        raw = quotes.get(dimension)
        if not raw:
            continue
        clip = _clip_quote(raw)
        if not clip:
            continue
        items.append(EvidenceItem(dimension=dimension, quote=clip))
    return items


def build_evaluation_output(
    *,
    session_id: str,
    analysis: "LLMAnalysisResponse",
    mastery_score: float,
    session_status: str,
    student_answer: str = "",
    key_concepts: Optional[List[str]] = None,
    next_question: Optional[str] = None,
    target_bloom_level: Optional[str] = None,
    demonstrated_bloom_level: Optional[str] = None,
    guessing_detected: Optional[bool] = None,
) -> "EvaluationOutput":
    """Build a validated :class:`EvaluationOutput` from an analysis result.

    Computes the ``specificity`` dimension deterministically, carries the
    analysis's ``guessing_detected`` flag, and grounds every dimension score
    with an evidence quote from the answer (EVAL-05). An empty answer yields
    no evidence and therefore fails construction (un-grounded scores rejected).
    """
    if guessing_detected is None:
        guessing_detected = bool(getattr(analysis, "guessing_detected", False))
    return EvaluationOutput(
        session_id=session_id,
        concept_coverage=analysis.concept_coverage,
        logical_coherence=analysis.logical_coherence,
        causal_reasoning=analysis.causal_reasoning,
        error_awareness=analysis.error_awareness,
        specificity=_specificity_score(student_answer, key_concepts),
        mastery_score=mastery_score,
        next_question=next_question,
        session_status=session_status,
        guessing_detected=guessing_detected,
        evidence=build_evidence(student_answer, key_concepts),
        target_bloom_level=target_bloom_level,
        demonstrated_bloom_level=demonstrated_bloom_level,
    )


