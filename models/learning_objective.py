"""LearningObjective model (F01 — Learning Objective Schema)."""

import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from bloom.taxonomy import BLOOM_LEVELS, KNOWLEDGE_TYPES, VERB_MAP

TEXT_MAX_CHARS = 200
VERB_START_MAX_WORDS = 3  # verb must appear within the first N words of text

NON_MEASURABLE_PHRASES = (
    "know",
    "be familiar with",
    "understand",
    "learn about",
    "be aware of",
    "grasp",
)


def _starts_with_verb_nearby(text: str, verb: str) -> bool:
    words = text.strip().split()[:VERB_START_MAX_WORDS]
    normalized_verb = verb.strip().lower()
    return any(re.sub(r"[^a-z]", "", w.lower()) == normalized_verb for w in words)


def _contains_non_measurable_phrase(text: str) -> Optional[str]:
    lower = text.lower()
    for phrase in NON_MEASURABLE_PHRASES:
        if lower.startswith(phrase):
            return phrase
    return None


class LearningObjective(BaseModel):
    """Model representing a single learning objective.

    Node mirror: `study-partner-api/study/src/validators/learningObjective.js`.
    Rules MUST stay identical on both sides. Depends on BLOOM-01
    (bloom/taxonomy.py) for enums and the verb map.

    Fields are snake_case (matching this codebase's convention, see
    models/task.py), with camelCase aliases so payloads shaped like the
    Node contract `{objectiveId, topicId, knowledgeType, bloomLevel, verb,
    text}` still populate correctly.

    Unlike the Node validator (which returns {valid, errors} and never
    throws), this model raises ValidationError on instantiation. Rejection
    logging therefore happens at the call site (wherever
    LearningObjective(**payload) is invoked), not inside this file — see
    learning_objective.usage_example.py.
    """

    objective_id: str = Field(
        ..., alias="objectiveId", description="Unique learning objective identifier"
    )
    topic_id: str = Field(..., alias="topicId", description="Topic this objective belongs to")
    knowledge_type: str = Field(
        ...,
        alias="knowledgeType",
        description=f"One of: {', '.join(KNOWLEDGE_TYPES)}",
    )
    bloom_level: str = Field(
        ..., alias="bloomLevel", description=f"One of: {', '.join(BLOOM_LEVELS)}"
    )
    verb: str = Field(..., description="Measurable action verb, must match bloom_level's verb map")
    text: str = Field(..., description="Objective text, max 200 chars, verb near the start")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "objectiveId": "obj_xyz789",
                "topicId": "topic_linear_equations",
                "knowledgeType": "conceptual",
                "bloomLevel": "apply",
                "verb": "Solve",
                "text": "Solve systems of linear equations using substitution.",
            }
        }

    @field_validator("objective_id", "topic_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v

    @field_validator("knowledge_type")
    @classmethod
    def _valid_knowledge_type(cls, v: str) -> str:
        if v not in KNOWLEDGE_TYPES:
            raise ValueError(f"knowledge_type must be one of: {', '.join(KNOWLEDGE_TYPES)}")
        return v

    @field_validator("bloom_level")
    @classmethod
    def _valid_bloom_level(cls, v: str) -> str:
        if v not in BLOOM_LEVELS:
            raise ValueError(f"bloom_level must be one of: {', '.join(BLOOM_LEVELS)}")
        return v

    @field_validator("text")
    @classmethod
    def _valid_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must be a non-empty string")
        if len(v) > TEXT_MAX_CHARS:
            raise ValueError(f"text exceeds {TEXT_MAX_CHARS} chars")
        vague = _contains_non_measurable_phrase(v)
        if vague:
            raise ValueError(f'text uses a non-measurable phrasing: "{vague}"')
        return v

    @model_validator(mode="after")
    def _verb_matches_level_and_text(self) -> "LearningObjective":
        allowed_verbs = VERB_MAP.get(self.bloom_level, ())
        if self.verb not in allowed_verbs:
            raise ValueError(
                f'verb must be one of: {", ".join(allowed_verbs)} for bloom_level "{self.bloom_level}"'
            )
        if not _starts_with_verb_nearby(self.text, self.verb):
            raise ValueError(f'verb "{self.verb}" must appear at/near the start of text')
        return self