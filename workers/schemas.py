"""Strict input schemas for AI workers (F02 / PLAN-02).

Mirrored by Node `shared/ai-messaging/payloadSchemas.js` — limits MUST stay
identical on both sides so the orchestrator rejects pre-publish exactly what
the worker would reject post-delivery.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Canonical limits (keep in sync with payloadSchemas.js)
GOAL_MAX_CHARS = 500
CONCEPTS_MAX_ITEMS = 50
CONCEPT_MAX_CHARS = 100
AVAILABLE_MINUTES_MIN = 1
AVAILABLE_MINUTES_MAX = 7 * 24 * 60  # one week

# BLOOM-10 weak-competency limits
WEAK_COMPETENCIES_MAX_ITEMS = 10
LEVEL_MAX_CHARS = 128

# Coach input limits (COACH-02) — keep in sync with payloadSchemas.js
COACH_SESSION_ID_MAX_CHARS = 64
COACH_MAX_SIGNALS = 20
COACH_MAX_MESSAGES = 40
COACH_MESSAGE_MAX_CHARS = 2000
COACH_MAX_PAYLOAD_BYTES = 16 * 1024  # 16 KB total payload cap

# Evaluator limits (EVAL-02) — keep in sync with payloadSchemas.js (backend)
EVAL_SESSION_ID_MAX_CHARS = 64
EVAL_STEP_MIN = 1
EVAL_ANSWER_MIN_CHARS = 1
EVAL_ANSWER_MAX_CHARS = 5000
EVAL_CONTEXT_ID_MAX_CHARS = 64
EVAL_OBJECTIVE_ID_MAX_CHARS = 64

# Knowledge extraction input limits (BLOOM-03) — keep in sync with payloadSchemas.js
COURSE_ID_MAX_CHARS = 64
DOCUMENT_ID_MAX_CHARS = 64
CONTENT_REF_MAX_CHARS = 256


class WeakCompetencyRequest(BaseModel):
    """One weak competency entry (BLOOM-10), from the Node competency profile.

    `scores` maps each lowercase bloom level -> 0..1 score. `unlocked_levels`
    lists the levels the progression gate (N-1 >= 0.7) currently permits.
    """

    model_config = ConfigDict(extra="forbid")

    topic_id: str = Field(..., max_length=LEVEL_MAX_CHARS)
    topic_title: Optional[str] = Field(default=None, max_length=LEVEL_MAX_CHARS)
    knowledge_type: Optional[str] = Field(default=None, max_length=32)
    scores: dict = Field(default_factory=dict)
    current_level: Optional[str] = Field(default=None, max_length=16)
    unlocked_levels: List[str] = Field(default_factory=list, max_length=6)


class PlannerRequest(BaseModel):
    """Validated payload for `study.plan.generate` jobs."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(..., min_length=1, max_length=GOAL_MAX_CHARS)
    concepts: List[str] = Field(default_factory=list, max_length=CONCEPTS_MAX_ITEMS)
    course_id: Optional[str] = Field(default=None, max_length=64)
    deadline: Optional[datetime] = None
    available_minutes: int = Field(
        default=120,
        ge=AVAILABLE_MINUTES_MIN,
        le=AVAILABLE_MINUTES_MAX,
    )
    # BLOOM-10: top-K weakest competencies (weakest first) for targeting.
    weak_competencies: List[WeakCompetencyRequest] = Field(default_factory=list)

    @field_validator("goal")
    @classmethod
    def _goal_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("goal must not be blank")
        return v

    @field_validator("concepts")
    @classmethod
    def _concepts_bounded(cls, v: List[str]) -> List[str]:
        for c in v:
            if not isinstance(c, str) or not c.strip():
                raise ValueError("each concept must be a non-empty string")
            if len(c) > CONCEPT_MAX_CHARS:
                raise ValueError(f"concept exceeds {CONCEPT_MAX_CHARS} chars")
        return v

    @field_validator("weak_competencies")
    @classmethod
    def _weak_competencies_bounded(cls, v: List[WeakCompetencyRequest]) -> List[WeakCompetencyRequest]:
        if len(v) > WEAK_COMPETENCIES_MAX_ITEMS:
            raise ValueError(f"weak_competencies exceeds {WEAK_COMPETENCIES_MAX_ITEMS} items")
        for wc in v:
            for level, score in wc.scores.items():
                if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                    raise ValueError(f"score for '{level}' must be in [0,1]")
        return v

    def to_planner_input(self, user_id: str):
        """Map onto the legacy PlannerInput consumed by PlannerAgent."""
        from datetime import datetime, timedelta, timezone

        from agents.planner.models.task_graph import PlannerInput, WeakCompetency

        DEFAULT_DEADLINE_DAYS = 7
        deadline = self.deadline or datetime.now(timezone.utc) + timedelta(
            days=DEFAULT_DEADLINE_DAYS
        )
        return PlannerInput(
            goal=self.goal,
            deadline_iso=deadline.isoformat(),
            available_minutes=self.available_minutes,
            user_id=user_id,
            retrieved_concepts=self.concepts or None,
            weak_competencies=[
                WeakCompetency(
                    topic_id=wc.topic_id,
                    topic_title=wc.topic_title,
                    knowledge_type=wc.knowledge_type,
                    scores=wc.scores,
                    current_level=wc.current_level,
                    unlocked_levels=tuple(wc.unlocked_levels),
                )
                for wc in self.weak_competencies
            ],
        )


class CoachSignal(BaseModel):
    """One ML signal reading from the recent window (mirrors SignalSnapshot,
    minus user_id — identity must come from the authenticated envelope only)."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    focus_state: Literal["Focused", "Drifting", "Lost"]
    focus_score: float = Field(ge=0.0, le=1.0)
    fatigue_state: Literal["Alert", "Moderate", "High", "Critical"]
    fatigue_score: float = Field(ge=0.0, le=1.0)
    focus_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fatigue_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    focus_trend: Optional[float] = None

    @field_validator(
        "focus_score",
        "fatigue_score",
        "focus_confidence",
        "fatigue_confidence",
        mode="before",
    )
    @classmethod
    def _no_boolean_score(cls, v):
        # lax mode otherwise coerces True->1.0; reject to match the JS edge
        if type(v) is bool:
            raise ValueError("must be a number, not a boolean")
        return v


class CoachMessage(BaseModel):
    """A recent chat message. User content is untrusted — isolated from coach
    instructions at prompt-build time (COACH-03)."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=COACH_MESSAGE_MAX_CHARS)

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank")
        return v


class CoachRequest(BaseModel):
    """Validated payload for `study.coach.nudge` jobs (COACH-02).

    Limits mirror `payloadSchemas.js` so the orchestrator rejects pre-publish
    exactly what the worker rejects post-delivery. `userId` is deliberately
    absent — it is only ever taken from the authenticated envelope.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: Optional[str] = Field(
        default=None, max_length=COACH_SESSION_ID_MAX_CHARS
    )
    signals: List[CoachSignal] = Field(default_factory=list, max_length=COACH_MAX_SIGNALS)
    messages: List[CoachMessage] = Field(default_factory=list, max_length=COACH_MAX_MESSAGES)
    focus_state: Optional[Literal["Focused", "Drifting", "Lost"]] = None
    focus_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fatigue_state: Optional[Literal["Alert", "Moderate", "High", "Critical"]] = None
    fatigue_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ignored_count: int = Field(default=0, ge=0)
    do_not_disturb: bool = Field(default=False, strict=True)
    current_time: Optional[datetime] = None

    @field_validator("focus_score", "fatigue_score", mode="before")
    @classmethod
    def _no_boolean_score(cls, v):
        # lax mode otherwise coerces True->1.0; reject to match the JS edge
        if type(v) is bool:
            raise ValueError("must be a number, not a boolean")
        return v

    @field_validator("ignored_count", mode="before")
    @classmethod
    def _no_boolean_count(cls, v):
        if type(v) is bool:
            raise ValueError("ignored_count must be an integer")
        return v

    @model_validator(mode="before")
    @classmethod
    def _payload_size_capped(cls, value):
        if isinstance(value, dict):
            size = len(json.dumps(value, separators=(",", ":")).encode("utf-8"))
            if size > COACH_MAX_PAYLOAD_BYTES:
                raise ValueError(
                    f"payload exceeds {COACH_MAX_PAYLOAD_BYTES} bytes "
                    f"(got {size})"
                )
        return value

    def to_coach_context(self) -> dict:
        """Flatten onto run_coach kwargs; user identity is injected by the
        worker from the envelope, never from the payload."""
        return {
            "current_time": self.current_time,
            "ignored_count": self.ignored_count,
            "do_not_disturb": self.do_not_disturb,
            "live_focus_score": self.focus_score,
            "live_focus_state": self.focus_state,
            "live_fatigue_score": self.fatigue_score,
            "live_fatigue_state": self.fatigue_state,
        }


class EvaluationRequest(BaseModel):
    """Validated payload for `study.eval.step` jobs (F04 / EVAL-02).

    Wire fields are camelCase to match the Node edge validator + publisher
    (``shared/ai-messaging/payloadSchemas.js`` / ``jobs.js``). ``step`` starts
    at 1 (creates the session from ``contextId``) and increments each answer
    turn; ``studentAnswer`` is the untrusted answer to process. ``userId`` is
    deliberately absent — it only ever comes from the authenticated envelope.

    `objectiveId` is accepted as OPTIONAL already (EVAL-08 carries it through to
    the per-step result feed so the backend can persist it when present); the F14
    Bloom learning-objective targeting logic that actually emits it is still
    deferred to a separate story.
    """

    model_config = ConfigDict(extra="forbid")

    sessionId: str = Field(..., min_length=1, max_length=EVAL_SESSION_ID_MAX_CHARS)
    step: int = Field(..., ge=EVAL_STEP_MIN, strict=True)
    contextId: str = Field(..., min_length=1, max_length=EVAL_CONTEXT_ID_MAX_CHARS)
    studentAnswer: str = Field(
        ...,
        min_length=EVAL_ANSWER_MIN_CHARS,
        max_length=EVAL_ANSWER_MAX_CHARS,
    )
    objectiveId: Optional[str] = Field(
        None,
        min_length=1,
        max_length=EVAL_OBJECTIVE_ID_MAX_CHARS,
        description="Learning-objective id (F14). Optional; persisted per step when present.",
    )
    # EVAL-02b: server-resolved target context.  Node resolves the objective's
    # bloomLevel + knowledgeType from the shared learning_objectives collection
    # (Python never touches Mongo) and carries them as camelCase wire fields in
    # the job payload so the worker can target the session's question depth.
    # Absent when objectiveId is absent (no targeting).
    targetBloomLevel: Optional[str] = Field(
        None,
        max_length=EVAL_OBJECTIVE_ID_MAX_CHARS,
        description="Bloom level echoed from the resolved learning objective.",
    )
    knowledgeType: Optional[str] = Field(
        None,
        max_length=EVAL_OBJECTIVE_ID_MAX_CHARS,
        description="Knowledge type from the resolved learning objective.",
    )

    @field_validator("sessionId", "contextId", "studentAnswer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        # len() enforces min_length, but reject whitespace-only values too
        if not v.strip():
            raise ValueError("field must not be blank")
        return v

    @field_validator("objectiveId")
    @classmethod
    def _objective_id_not_blank(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("objectiveId must not be blank when provided")
        return v

    @property
    def session_id(self) -> str:
        return self.sessionId

    @property
    def context_id(self) -> str:
        return self.contextId

    @property
    def student_answer(self) -> str:
        return self.studentAnswer


class KnowledgeExtractRequest(BaseModel):
    """Validated payload for `study.knowledge.extract` jobs (BLOOM-03).

    Input contract `{documentId, courseId, contentRef}` — raw content is loaded
    from storage by the worker, never inline in the envelope, so only reference
    fields are required. Limits mirror `payloadSchemas.js` so the orchestrator
    rejects pre-publish exactly what the worker rejects post-delivery.
    """

    model_config = ConfigDict(extra="forbid")

    documentId: str = Field(..., min_length=1, max_length=DOCUMENT_ID_MAX_CHARS)
    courseId: str = Field(..., min_length=1, max_length=COURSE_ID_MAX_CHARS)
    contentRef: str = Field(..., min_length=1, max_length=CONTENT_REF_MAX_CHARS)

    @field_validator("documentId", "courseId", "contentRef")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v
