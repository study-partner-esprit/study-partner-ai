"""Contract tests for AI messaging envelopes (AI-COM-02/03, TEST-04).

These mirror tests/shared/ai-envelope.test.js on the Node side. Both suites
must stay green for the same fixtures — schema drift between Node and Python
fails CI.
"""

import pytest
from pydantic import ValidationError

from messaging.envelope import (
    ENVELOPE_VERSION,
    AI_JOB_TYPES,
    validate_job_envelope,
    validate_result_envelope,
)

UUID_A = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
UUID_B = "0f8e2d1a-3b4c-4d6e-8f80-91a2b3c4d5e6"

VALID_BASE = {
    "messageId": UUID_A,
    "correlationId": UUID_B,
    "type": "study.plan.generate",
    "version": ENVELOPE_VERSION,
    "requestId": "req-abc123",
    "timestamp": "2026-08-19T08:00:00Z",
}


class TestJobEnvelope:
    def test_accepts_well_formed_job(self):
        job = validate_job_envelope(
            {**VALID_BASE, "userId": "user-1", "payload": {"goal": "learn graphs"}}
        )
        assert job.userId == "user-1"

    @pytest.mark.parametrize("type_", sorted(AI_JOB_TYPES))
    def test_accepts_all_registered_types(self, type_):
        validate_job_envelope({**VALID_BASE, "type": type_, "userId": "u", "payload": {}})

    def test_rejects_unknown_type(self):
        with pytest.raises(ValidationError):
            validate_job_envelope(
                {
                    **VALID_BASE,
                    "type": "study.plan.delete-everything",
                    "userId": "u",
                    "payload": {},
                }
            )

    @pytest.mark.parametrize("userId", [None, "", "x" * 200])
    def test_rejects_invalid_user_id(self, userId):
        with pytest.raises(ValidationError):
            validate_job_envelope({**VALID_BASE, "userId": userId, "payload": {}})

    def test_rejects_non_object_payload(self):
        with pytest.raises(ValidationError):
            validate_job_envelope({**VALID_BASE, "userId": "u", "payload": [1]})

    def test_rejects_wrong_version(self):
        with pytest.raises(ValidationError):
            validate_job_envelope({**VALID_BASE, "version": "2", "userId": "u", "payload": {}})

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            validate_job_envelope(
                {**VALID_BASE, "userId": "u", "payload": {}, "status": "PENDING"}
            )


class TestResultEnvelope:
    def test_accepts_completed_result(self):
        result = validate_result_envelope({**VALID_BASE, "status": "completed", "payload": {}})
        assert result.status == "completed"

    def test_accepts_sanitized_failure(self):
        result = validate_result_envelope(
            {**VALID_BASE, "status": "failed", "error": "LLM provider timeout after retries"}
        )
        assert result.error == "LLM provider timeout after retries"

    @pytest.mark.parametrize(
        "error",
        [
            "Error: x\n    at handler (/app/src/x.py:10:5)",
            "auth failed: mongodb://admin:s3cret@mongo:27017",
            "mongodb+srv://user:pass@cluster.example.net/db",
        ],
    )
    def test_rejects_leaky_errors(self, error):
        with pytest.raises(ValidationError):
            validate_result_envelope({**VALID_BASE, "status": "failed", "error": error})

    def test_failure_requires_error_within_limits(self):
        with pytest.raises(ValidationError):
            validate_result_envelope({**VALID_BASE, "status": "failed"})
        with pytest.raises(ValidationError):
            validate_result_envelope({**VALID_BASE, "status": "failed", "error": "x" * 600})

    def test_completed_must_not_carry_error(self):
        with pytest.raises(ValidationError):
            validate_result_envelope(
                {**VALID_BASE, "status": "completed", "payload": {}, "error": "nope"}
            )
