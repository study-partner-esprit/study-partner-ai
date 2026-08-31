"""
Pydantic models for evaluation data structures and validation.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime, timezone


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


# ============================================================================
# EVALUATION OUTPUT (EVAL-04)
# ============================================================================

_DIM_BOUNDS = dict(ge=0.0, le=1.0)


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
) -> "EvaluationOutput":
    """Build a validated :class:`EvaluationOutput` from an analysis result.

    Computes the ``specificity`` dimension deterministically and clamps all
    scores into [0.0, 1.0]; out-of-range or extra fields fail construction
    via the schema's strict bounds.
    """
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
        target_bloom_level=target_bloom_level,
        demonstrated_bloom_level=demonstrated_bloom_level,
    )


