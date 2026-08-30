"""COACH-08 AC#1 — LLM timeout/quota failures are retryable, never mocked.

`call_gemini` degrades to the mock responder ONLY when the coach LLM is
unavailable by configuration (mock mode / no provider key). Real transient
infrastructure failures raise RetryableError so the coach job-bus retry policy
(AI-COM-06) owns recovery — CoachWorker retries, and falls back to the rule
engine on the final attempt.
"""

import json
from unittest.mock import patch

import pytest

from messaging.failures import RetryableError
from utils.llm_client import LLMRequestError, MissingMockResponderError

from agents.coach.decision import llm_decider


@patch("agents.coach.decision.llm_decider.ask")
def test_llm_request_error_propagates_as_retryable(mock_ask):
    mock_ask.side_effect = LLMRequestError("upstream timeout after 30s")

    with pytest.raises(RetryableError):
        llm_decider.call_gemini("system", "user", trace_id="tr-1")


@patch("agents.coach.decision.llm_decider.ask")
def test_missing_mock_responder_degrades_to_mock(mock_ask):
    mock_ask.side_effect = MissingMockResponderError("no key for agent coach")

    raw = llm_decider.call_gemini("system", "user", trace_id="tr-2")
    # Mock output is a JSON CoachAction the decider can parse.
    assert json.loads(raw)