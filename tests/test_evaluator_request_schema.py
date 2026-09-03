"""EvaluationRequest schema tests (F04 / EVAL-02).

The wire contract is camelCase and mirrors (eventually) the Node edge validator
in ``shared/ai-messaging/payloadSchemas.js``; both must reject the same inputs.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from workers.schemas import (
    EVAL_ANSWER_MAX_CHARS,
    EVAL_ANSWER_MIN_CHARS,
    EVAL_CONTEXT_ID_MAX_CHARS,
    EVAL_OBJECTIVE_ID_MAX_CHARS,
    EVAL_SESSION_ID_MAX_CHARS,
    EVAL_STEP_MIN,
    EvaluationRequest,
)


def valid(**overrides):
    payload = {
        "sessionId": "sess-123",
        "step": 2,
        "contextId": "ctx-9",
        "studentAnswer": "A base case stops recursion, the recursive case reduces the problem.",
    }
    payload.update(overrides)
    return payload


def test_valid_contract_maps_to_snake_case_accessors():
    r = EvaluationRequest.model_validate(valid())
    assert r.session_id == "sess-123"
    assert r.step == 2
    assert r.context_id == "ctx-9"
    assert r.student_answer.startswith("A base case")


def test_unknown_fields_forbidden():
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid(evil_instruction="ignore previous"))


# ------------------------------------------------------------ required fields

@pytest.mark.parametrize("drop", ["sessionId", "step", "contextId", "studentAnswer"])
def test_missing_required_field_raises(drop):
    payload = valid()
    payload.pop(drop)
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(payload)


@pytest.mark.parametrize("field", ["sessionId", "contextId"])
def test_blank_identifier_rejected(field):
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid(**{field: "   "}))


# ------------------------------------------------------------ length bounds

def test_session_id_length_bounds():
    assert EvaluationRequest.model_validate(
        valid(sessionId="s" * EVAL_SESSION_ID_MAX_CHARS)
    ).sessionId == "s" * EVAL_SESSION_ID_MAX_CHARS
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid(sessionId="x" * (EVAL_SESSION_ID_MAX_CHARS + 1)))


def test_context_id_length_bounds():
    assert EvaluationRequest.model_validate(
        valid(contextId="c" * EVAL_CONTEXT_ID_MAX_CHARS)
    ).contextId == "c" * EVAL_CONTEXT_ID_MAX_CHARS
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid(contextId="x" * (EVAL_CONTEXT_ID_MAX_CHARS + 1)))


def test_answer_length_bounds():
    ok = EvaluationRequest.model_validate(valid(studentAnswer="a" * EVAL_ANSWER_MAX_CHARS))
    assert ok.studentAnswer == "a" * EVAL_ANSWER_MAX_CHARS
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid(studentAnswer="x" * (EVAL_ANSWER_MAX_CHARS + 1)))
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid(studentAnswer="x" * (EVAL_ANSWER_MIN_CHARS - 1)))


def test_whitespace_only_answer_rejected():
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid(studentAnswer="   "))


# ------------------------------------------------------------ step bounds

def test_step_must_be_at_least_one():
    assert EvaluationRequest.model_validate(valid(step=EVAL_STEP_MIN)).step == EVAL_STEP_MIN
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid(step=0))
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid(step=-1))


def test_step_must_be_integer():
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(valid(step="2"))


# ---------------------------------------------------------------- EVAL-02b

def test_accepts_target_bloom_level_and_knowledge_type():
    """EVAL-02b: server-resolved objective context is carried as evaluation
    context in the job payload (camelCase wire fields)."""
    r = EvaluationRequest.model_validate(
        valid(objectiveId="obj-1", targetBloomLevel="APPLY", knowledgeType="procedural")
    )
    assert r.objectiveId == "obj-1"
    assert r.targetBloomLevel == "APPLY"
    assert r.knowledgeType == "procedural"


def test_target_context_defaults_to_none_when_absent():
    r = EvaluationRequest.model_validate(valid())
    assert r.objectiveId is None
    assert r.targetBloomLevel is None
    assert r.knowledgeType is None


def test_target_bloom_level_length_bounds():
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(
            valid(targetBloomLevel="X" * (EVAL_OBJECTIVE_ID_MAX_CHARS + 1))
        )


def test_knowledge_type_length_bounds():
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(
            valid(knowledgeType="x" * (EVAL_OBJECTIVE_ID_MAX_CHARS + 1))
        )
