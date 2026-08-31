"""Validation / rejection pipeline tests (F04 / EVAL-06).

Verifies:
- reject missing required fields (score, strengths, weaknesses)
- reject out-of-range scores (< 0.0 or > 1.0)
- reject non-numeric score
- reject missing score (None)
- reject empty student answer (no grounding)
- reject empty strengths AND weaknesses
- accept valid parsed output
- validate_question_output rejects incoherent questions
- build_failure_analysis returns neutral fallback
- agent retries once then FAILED with sanitized reason
"""

from __future__ import annotations

import logging
from unittest.mock import patch, MagicMock

import pytest

from agents.evaluator.validation import (
    validate_analysis_output,
    validate_question_output,
    build_failure_analysis,
)


# ---------------------------------------------------------------------------
# validate_analysis_output tests
# ---------------------------------------------------------------------------

class TestValidateAnalysisOutput:
    def test_valid_output(self):
        parsed = {
            "score": 0.75,
            "strengths": "Good understanding.",
            "weaknesses": "Missing some detail.",
            "missing_concepts": ["plasmalemma"],
        }
        ok, reason = validate_analysis_output(parsed, "The answer is osmosis.")
        assert ok is True
        assert reason == ""

    def test_missing_score(self):
        parsed = {"strengths": "ok", "weaknesses": "no"}
        ok, reason = validate_analysis_output(parsed, "answer")
        assert ok is False
        assert "score" in reason

    def test_missing_strengths_and_weaknesses(self):
        parsed = {"score": 0.5}
        ok, reason = validate_analysis_output(parsed, "answer")
        assert ok is False
        assert "missing" in reason.lower()

    def test_score_none(self):
        parsed = {"score": None, "strengths": "ok", "weaknesses": "no"}
        ok, reason = validate_analysis_output(parsed, "answer")
        assert ok is False
        assert "could not be extracted" in reason

    def test_score_out_of_range_high(self):
        parsed = {
            "score": 1.5,
            "strengths": "ok",
            "weaknesses": "no",
        }
        ok, reason = validate_analysis_output(parsed, "answer")
        assert ok is False
        assert "out of range" in reason

    def test_score_out_of_range_low(self):
        parsed = {
            "score": -0.1,
            "strengths": "ok",
            "weaknesses": "no",
        }
        ok, reason = validate_analysis_output(parsed, "answer")
        assert ok is False
        assert "out of range" in reason

    def test_score_not_numeric(self):
        parsed = {
            "score": "not a number",
            "strengths": "ok",
            "weaknesses": "no",
        }
        ok, reason = validate_analysis_output(parsed, "answer")
        assert ok is False
        assert "not numeric" in reason

    def test_empty_answer_rejected(self):
        parsed = {
            "score": 0.5,
            "strengths": "ok",
            "weaknesses": "no",
        }
        ok, reason = validate_analysis_output(parsed, "")
        assert ok is False
        assert "too short" in reason

    def test_empty_strengths_and_weaknesses_rejected(self):
        parsed = {
            "score": 0.5,
            "strengths": "",
            "weaknesses": "",
        }
        ok, reason = validate_analysis_output(parsed, "answer")
        assert ok is False
        assert "empty strengths" in reason

    def test_whitespace_only_answer_rejected(self):
        parsed = {
            "score": 0.5,
            "strengths": "ok",
            "weaknesses": "no",
        }
        ok, reason = validate_analysis_output(parsed, "   ")
        assert ok is False

    def test_non_dict_rejected(self):
        ok, reason = validate_analysis_output("not a dict", "answer")
        assert ok is False
        assert "non-dict" in reason

    def test_score_boundary_0(self):
        parsed = {"score": 0.0, "strengths": "ok", "weaknesses": "no"}
        ok, _ = validate_analysis_output(parsed, "answer")
        assert ok is True

    def test_score_boundary_1(self):
        parsed = {"score": 1.0, "strengths": "ok", "weaknesses": "no"}
        ok, _ = validate_analysis_output(parsed, "answer")
        assert ok is True


# ---------------------------------------------------------------------------
# validate_question_output tests
# ---------------------------------------------------------------------------

class TestValidateQuestionOutput:
    def test_valid_question(self):
        ok, reason = validate_question_output(
            "What is osmosis and why is it important in this context?"
        )
        assert ok is True

    def test_empty_question_rejected(self):
        ok, reason = validate_question_output("")
        assert ok is False
        assert "empty" in reason

    def test_none_question_rejected(self):
        ok, reason = validate_question_output(None)
        assert ok is False


# ---------------------------------------------------------------------------
# build_failure_analysis tests
# ---------------------------------------------------------------------------

class TestBuildFailureAnalysis:
    def test_returns_neutral_analysis(self):
        analysis, mastery = build_failure_analysis("answer", "missing fields")
        assert analysis.concept_coverage == 0.3
        assert mastery == 0.3
        assert "missing fields" in analysis.answer_feedback

    def test_mastery_capped_at_0_3(self):
        _, mastery = build_failure_analysis("answer", "test")
        assert mastery == 0.3


# ---------------------------------------------------------------------------
# Agent retry + rejection logging
# ---------------------------------------------------------------------------

class TestAgentValidationRetry:
    def test_retry_on_invalid_then_success(self):
        from agents.evaluator.agent import EvaluatorAgent

        agent = EvaluatorAgent(require_llm=False)
        session_id = agent.start_session(
            task_title="Osmosis",
            task_description="diffusion",
            task_details="Osmosis is movement of water across semipermeable membrane. Key: osmosis, semipermeable.",
        )["session_id"]

        # First call returns invalid, second returns valid
        invalid_parsed = {"score": None, "strengths": "", "weaknesses": ""}
        valid_parsed = {
            "score": 0.6,
            "strengths": "Good point.",
            "weaknesses": "Needs detail.",
            "missing_concepts": [],
        }
        call_count = 0
        def mock_generate(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Score: invalid"
            return "Score: 0.6\nStrengths: Good point.\nWeaknesses: Needs detail."

        agent.llm = MagicMock()
        agent.llm.generate = mock_generate

        with patch("agents.evaluator.agent.parse_analysis_response") as mock_parse:
            mock_parse.side_effect = [invalid_parsed, valid_parsed]
            result = agent.handle_user_answer(session_id, "Osmosis moves water.")

        assert "evaluation_output" in result
        assert call_count == 2  # retry happened

    def test_double_failure_returns_neutral_score(self):
        from agents.evaluator.agent import EvaluatorAgent

        agent = EvaluatorAgent(require_llm=False)
        session_id = agent.start_session(
            task_title="Osmosis",
            task_description="diffusion",
            task_details="Osmosis is movement of water. Key: osmosis.",
        )["session_id"]

        bad_parsed = {"score": None, "strengths": "", "weaknesses": ""}

        agent.llm = MagicMock()
        agent.llm.generate.return_value = "Score: invalid"

        with patch("agents.evaluator.agent.parse_analysis_response", return_value=bad_parsed):
            with patch("agents.evaluator.agent.validate_analysis_output", return_value=(False, "no score")):
                result = agent.handle_user_answer(session_id, "answer")

        out = result["evaluation_output"]
        assert out["mastery_score"] <= 0.3
        assert "no score" in result.get("feedback", "") or "skipped" in result.get("feedback", "").lower()

    def test_rejection_logged(self, caplog):
        from agents.evaluator.agent import EvaluatorAgent

        agent = EvaluatorAgent(require_llm=False)
        session_id = agent.start_session(
            task_title="Osmosis",
            task_description="diffusion",
            task_details="Osmosis is movement of water. Key: osmosis.",
        )["session_id"]

        bad_parsed = {"score": None, "strengths": "", "weaknesses": ""}

        agent.llm = MagicMock()
        agent.llm.generate.return_value = "raw LLM text here"

        with patch("agents.evaluator.agent.parse_analysis_response", return_value=bad_parsed):
            with patch("agents.evaluator.agent.validate_analysis_output", return_value=(False, "missing score")):
                with caplog.at_level(logging.WARNING):
                    agent.handle_user_answer(session_id, "answer")

        assert "EVAL-06" in caplog.text
        assert "raw LLM text here" in caplog.text
