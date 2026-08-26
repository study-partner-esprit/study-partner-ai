"""Strict input schemas for AI workers (F02 / PLAN-02).

Mirrored by Node `shared/ai-messaging/payloadSchemas.js` — limits MUST stay
identical on both sides so the orchestrator rejects pre-publish exactly what
the worker would reject post-delivery.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Canonical limits (keep in sync with payloadSchemas.js)
GOAL_MAX_CHARS = 500
CONCEPTS_MAX_ITEMS = 50
CONCEPT_MAX_CHARS = 100
AVAILABLE_MINUTES_MIN = 1
AVAILABLE_MINUTES_MAX = 7 * 24 * 60  # one week


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
