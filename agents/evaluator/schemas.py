"""
Pydantic models for evaluation data structures and validation.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
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
