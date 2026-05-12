"""
Pydantic models for evaluation data structures and validation.
"""

from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class CourseLevel(str, Enum):
    """Course difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


# ============================================================================
# SUBTOPIC MODEL
# ============================================================================

class SubTopic(BaseModel):
    """A subtopic with key concepts."""
    
    id: str = Field(..., description="Subtopic ID or name")
    title: str = Field(..., description="Subtopic title")
    summary: Optional[str] = Field(None, description="Subtopic summary")
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts in this subtopic")


# ============================================================================
# TASK EVALUATION CONTEXT
# ============================================================================

class TaskEvaluationContext(BaseModel):
    """Context for task evaluation - built in-memory from user input."""

    task_title: str = Field(..., description="Title of the task")
    task_description: str = Field(..., description="Description of the task")
    task_details: str = Field(..., description="Detailed explanation of the task")
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts to evaluate")


# ============================================================================
# QUESTION GENERATION RESPONSE
# ============================================================================

class LLMQuestionResponse(BaseModel):
    """LLM response for Socratic question generation."""

    question: str = Field(..., description="The generated Socratic question")
    depth_level: str = Field(
        ...,
        description="Depth level: what, why, or how"
    )
    concept_focus: str = Field(
        ...,
        description="Which concept this question targets"
    )


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
# REWARD PAYLOAD
# ============================================================================

class RewardPayload(BaseModel):
    """Reward given when mastery is confirmed."""

    learning_points: int = Field(..., description="Points earned for mastery")
    streak_increment: int = Field(..., description="Streak increment (usually 1)")
    concepts_covered: List[str] = Field(
        default_factory=list,
        description="Concepts demonstrated"
    )


# ============================================================================
# RESCHEDULE PAYLOAD
# ============================================================================

class ReschedulePayload(BaseModel):
    """Reschedule recommendation when evaluation fails."""

    action: str = Field(
        ...,
        description="Action: REVIEW, SIMPLIFY, or BREAK_DOWN"
    )
    reason: str = Field(..., description="Why this recommendation")
    weak_concepts: List[str] = Field(
        default_factory=list,
        description="Concepts that need improvement"
    )
    misconceptions: List[str] = Field(
        default_factory=list,
        description="Misconceptions to address"
    )


# ============================================================================
# EVALUATION RESULT
# ============================================================================

class EvaluationResult(BaseModel):
    """Result of an evaluation step."""

    state: EvaluationState
    mastery_score: float = Field(..., ge=0.0, le=1.0)
    questions_asked: int
    feedback: Optional[str] = None
    next_question: Optional[str] = None
    reward: Optional[RewardPayload] = None
    reschedule: Optional[ReschedulePayload] = None


# ============================================================================
# SESSION MODEL
# ============================================================================
from enum import Enum as _Enum


class SessionState(str, _Enum):
    """States for interactive evaluation session."""

    ASKING = "asking"
    ANALYZING = "analyzing"
    COMPLETE = "complete"


class EvaluationSession(BaseModel):
    """Represents an interactive evaluation session - fully in-memory."""

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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# AUDIO INPUT
# ============================================================================

class AudioInput(BaseModel):
    """Audio input from student."""

    audio_file_path: str = Field(..., description="Path to audio file")
    format: str = Field(default="wav", description="Audio format: wav, mp3, m4a, etc.")
    language: Optional[str] = Field(default=None, description="Language code (auto-detect if None)")


class TextInput(BaseModel):
    """Text input from student."""

    text: str = Field(..., description="Student's text answer")


# ============================================================================
# STUDENT INPUT (UNION TYPE)
# ============================================================================

StudentInput = AudioInput | TextInput
