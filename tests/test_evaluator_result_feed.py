"""EVAL-08 result-feed tests (F04) — no broker, no Mongo.

The ai.results payload is the ONLY path to MongoDB: the Python backend never
writes to Mongo; the Node backend persists whatever arrives. These tests pin
that the payload is a complete, queryable per-step record — top-level
``sessionId`` / ``step``, ``demonstratedBloomLevel`` (the raw feed for BLOOM-08)
and an optional ``objectiveId`` — while staying additive (backward-compatible)
so existing consumers like ``AiJob.completeByCorrelation`` are unaffected.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from workers.evaluator_worker import EvaluatorWorker
from workers.idempotency import InMemoryIdempotencyStore
from workers.schemas import EvaluationRequest
from workers.session_store import InMemorySessionStore


def envelope(payload: dict) -> dict:
    return {
        "messageId": str(uuid.uuid4()),
        "correlationId": "9b1e2f3c-4d5e-4f60-8a71-b2c3d4e5f607",
        "type": "study.eval.step",
        "version": "1",
        "userId": "user-42",
        "requestId": "req-eval08",
        "timestamp": "2026-08-30T09:00:00Z",
        "payload": payload,
    }


def make_worker() -> EvaluatorWorker:
    from agents.evaluator.agent import EvaluatorAgent

    return EvaluatorWorker(
        idempotency_store=InMemoryIdempotencyStore(),
        session_store=InMemorySessionStore(),
        agent=EvaluatorAgent(require_llm=False),
    )


def start_payload(
    session_id: str = "sess-eval08",
    answer: str = "Recursion needs a base case that stops it and a recursive case that reduces the problem.",
    objective_id: str | None = None,
) -> dict:
    payload = {
        "sessionId": session_id,
        "step": 1,
        "contextId": "ctx-recursion",
        "studentAnswer": answer,
    }
    if objective_id is not None:
        payload["objectiveId"] = objective_id
    return payload


# ----------------------------------------------------------- per-step record

async def test_handle_returns_carrier_fields_on_published_record():
    worker = make_worker()
    result = await worker.handle(start_payload("sess-feed"), envelope(None))

    assert result["sessionId"] == "sess-feed"
    assert result["step"] == 1
    assert "demonstratedBloomLevel" in result
    assert "evaluation_output" in result


async def test_record_keeps_existing_fields_intact():
    worker = make_worker()
    result = await worker.handle(start_payload("sess-compat"), envelope(None))

    # EVAL-08 is additive only — every pre-existing consumer-facing key survives.
    assert result["session_id"] == "sess-compat"
    assert "mastery_score" in result
    assert "state" in result
    assert "feedback" in result
    assert result["evaluation_output"]["session_status"] == result["state"]


def test_with_step_record_lifts_non_null_bloom_level():
    worker = make_worker()
    request = EvaluationRequest.model_validate(start_payload())
    result = {
        "session_id": "sess-bloom",
        "state": "CONTINUE",
        "mastery_score": 0.8,
        "evaluation_output": {
            "mastery_score": 0.8,
            "demonstrated_bloom_level": "ANALYZE",
            "next_question": "Why does that base case stop the recursion?",
        },
    }

    record = worker._with_step_record(result, request)

    assert record["demonstratedBloomLevel"] == "ANALYZE"
    assert record["sessionId"] == request.sessionId
    assert record["step"] == request.step
    assert "objectiveId" not in record


def test_with_step_record_reflects_null_bloom_level():
    worker = make_worker()
    request = EvaluationRequest.model_validate(start_payload())
    result = {
        "session_id": "sess-nullbloom",
        "state": "CONTINUE",
        "mastery_score": 0.5,
        "evaluation_output": {"demonstrated_bloom_level": None, "next_question": "..."},
    }

    record = worker._with_step_record(result, request)

    assert record["demonstratedBloomLevel"] is None


def test_with_step_record_adds_objective_id_when_present():
    worker = make_worker()
    request = EvaluationRequest.model_validate(
        start_payload(objective_id="obj-recursion-42")
    )
    result = {"session_id": "sess-obj", "state": "CONTINUE", "mastery_score": 0.7}

    record = worker._with_step_record(result, request)

    assert record["objectiveId"] == "obj-recursion-42"


def test_with_step_record_skips_objective_id_when_absent():
    worker = make_worker()
    request = EvaluationRequest.model_validate(start_payload())
    record = worker._with_step_record(
        {"session_id": "sess-noobj", "state": "FAILED", "mastery_score": 0.2}, request
    )
    assert "objectiveId" not in record


def test_with_step_record_ignores_non_dict_result():
    worker = make_worker()
    request = EvaluationRequest.model_validate(start_payload())
    assert worker._with_step_record(None, request) is None


# ---------------------------------------------------- objectiveId wire schema

def test_request_accepts_optional_objective_id():
    request = EvaluationRequest.model_validate(start_payload(objective_id="obj-1"))
    assert request.objectiveId == "obj-1"
    assert request.model_dump()["objectiveId"] == "obj-1"


def test_request_accepts_missing_objective_id():
    request = EvaluationRequest.model_validate(start_payload())
    assert request.objectiveId is None


def test_request_rejects_blank_objective_id():
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(start_payload(objective_id="   "))


def test_request_rejects_overlong_objective_id():
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(start_payload(objective_id="o" * 65))