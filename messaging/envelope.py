"""AI message contract — canonical envelopes (AI-COM-02 / AI-COM-03).

Python mirror of `study-partner-api/shared/ai-messaging/envelope.js`.
Both sides validate on publish AND on consume. Any schema change must be
applied to both files and to docs/contracts/ai-message-contract.md in the
same commit (enforced by the TEST-04 contract tests).
"""

import re
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

ENVELOPE_VERSION = "1"

AI_JOB_TYPES = frozenset(
    [
        "study.plan.generate",
        "study.coach.nudge",
        "study.eval.step",
        "study.search.query",
        "study.ingest.course",
        "study.schedule.apply",
    ]
)

RESULT_STATUSES = frozenset(["completed", "failed"])

_UUID_V4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Sanitized errors: no stack frames, no DB connection strings.
_UNSAFE_ERROR_RE = re.compile(r"stack|at .*\(|mongodb(\+srv)?://", re.IGNORECASE)


class _EnvelopeBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messageId: str
    correlationId: str
    type: str
    version: str
    requestId: str
    timestamp: datetime

    @field_validator("messageId", "correlationId")
    @classmethod
    def _check_uuid(cls, v: str) -> str:
        if not _UUID_V4_RE.match(v):
            raise ValueError("must be a UUID v4 string")
        return v

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if v not in AI_JOB_TYPES:
            raise ValueError(f"type must be one of: {', '.join(sorted(AI_JOB_TYPES))}")
        return v

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if v != ENVELOPE_VERSION:
            raise ValueError(f'version must be "{ENVELOPE_VERSION}"')
        return v

    @field_validator("requestId")
    @classmethod
    def _check_request_id(cls, v: str) -> str:
        if not v or len(v) > 128:
            raise ValueError("requestId must be a non-empty string (max 128 chars)")
        return v


class AiJobEnvelope(_EnvelopeBase):
    """Request envelope published by the Node orchestrator (AI-COM-02)."""

    model_config = ConfigDict(extra="forbid")

    userId: str
    payload: dict[str, Any]

    @field_validator("userId")
    @classmethod
    def _check_user_id(cls, v: str) -> str:
        # Never accept userId from a client body; it is set by Node from the JWT.
        if not v or len(v) > 128:
            raise ValueError("userId must be a non-empty string taken from the authenticated context")
        return v


class AiResultEnvelope(_EnvelopeBase):
    """Result event envelope published by Python workers (AI-COM-03)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "failed"]
    payload: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    @model_validator(mode="after")
    def _check_status_payload_error(self) -> "AiResultEnvelope":
        if self.status == "completed":
            if not isinstance(self.payload, dict):
                raise ValueError("completed results must carry a payload object")
            if self.error is not None:
                raise ValueError("completed results must not carry an error field")
        else:  # failed
            if not self.error or not self.error.strip():
                raise ValueError("failed results must carry a sanitized error message")
            if len(self.error) > 512:
                raise ValueError("error must be at most 512 chars")
            if _UNSAFE_ERROR_RE.search(self.error):
                raise ValueError(
                    "error must be sanitized: no stack traces or connection strings"
                )
        return self


def validate_job_envelope(message: Any) -> AiJobEnvelope:
    """Parse + validate an incoming job message. Raises ValidationError."""
    return AiJobEnvelope.model_validate(message)


def validate_result_envelope(message: Any) -> AiResultEnvelope:
    """Parse + validate an incoming result event. Raises ValidationError."""
    return AiResultEnvelope.model_validate(message)
