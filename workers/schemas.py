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

# Coach input limits (COACH-02) — keep in sync with payloadSchemas.js
COACH_SESSION_ID_MAX_CHARS = 64
COACH_MAX_SIGNALS = 20
COACH_MAX_MESSAGES = 40
COACH_MESSAGE_MAX_CHARS = 2000
COACH_MAX_PAYLOAD_BYTES = 16 * 1024  # 16 KB total payload cap

# Knowledge extraction input limits (BLOOM-03) — keep in sync with payloadSchemas.js
COURSE_ID_MAX_CHARS = 64
DOCUMENT_ID_MAX_CHARS = 64
CONTENT_REF_MAX_CHARS = 256


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

    def to_planner_input(self, user_id: str):
        """Map onto the legacy PlannerInput consumed by PlannerAgent."""
        from datetime import datetime, timedelta, timezone

        from agents.planner.models.task_graph import PlannerInput

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
