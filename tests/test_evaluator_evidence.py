"""Evidence grounding tests for EVAL-05.

Verifies:
- EvidenceItem schema: {dimension, quote}, quote 1–200 chars, no extras
- EvaluationOutput requires non-empty evidence list
- build_evidence extracts per-dimension quotes from the answer
- Empty / absent answer yields no evidence → ValidationError (un-grounded)
- guessing_detected is carried through from analysis to output
- Mastery scoring formula retained (smoothing guard) via MasteryScorer
"""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from agents.evaluator.schemas import (
    EvidenceItem,
    EvaluationOutput,
    LLMAnalysisResponse,
    build_evidence,
    build_evaluation_output,
)
from agents.evaluator.scoring import MasteryScorer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _analysis(**overrides) -> LLMAnalysisResponse:
    base = dict(
        concept_coverage=0.8,
        logical_coherence=0.7,
        causal_reasoning=0.6,
        error_awareness=0.5,
        answer_feedback="Good understanding.",
        guessing_detected=False,
        missing_concepts=[],
        misconceptions=[],
    )
    base.update(overrides)
    return LLMAnalysisResponse(**base)


_LONG_ANSWER = (
    "Osmosis is the movement of water across a semipermeable membrane "
    "toward higher solute concentration. This process occurs because the "
    "membrane allows water but not solute to pass. The result is an increase "
    "in turgor pressure inside the cell, which supports plant structure."
)


def _valid_output(**overrides) -> EvaluationOutput:
    kwargs = dict(
        session_id="s1",
        concept_coverage=0.8,
        logical_coherence=0.7,
        causal_reasoning=0.6,
        error_awareness=0.5,
        specificity=0.4,
        mastery_score=0.6,
        next_question=None,
        session_status="CONTINUE",
        guessing_detected=False,
        evidence=[EvidenceItem(dimension="concept_coverage", quote="osmosis")],
    )
    kwargs.update(overrides)
    return EvaluationOutput(**kwargs)


# ---------------------------------------------------------------------------
# EvidenceItem schema tests
# ---------------------------------------------------------------------------

class TestEvidenceItemSchema:
    def test_valid(self):
        e = EvidenceItem(dimension="concept_coverage", quote="Water moves by osmosis.")
        assert e.dimension == "concept_coverage"
        assert e.quote == "Water moves by osmosis."

    def test_empty_quote_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceItem(dimension="x", quote="")

    def test_long_quote_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceItem(dimension="x", quote="a" * 201)

    def test_200_char_quote_accepted(self):
        e = EvidenceItem(dimension="x", quote="a" * 200)
        assert len(e.quote) == 200

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceItem(dimension="x", quote="ok", bad=True)


# ---------------------------------------------------------------------------
# EvaluationOutput evidence + guessing_detected tests
# ---------------------------------------------------------------------------

class TestEvaluationOutputEvidence:
    def test_empty_evidence_list_rejected(self):
        with pytest.raises(ValidationError):
            _valid_output(evidence=[])

    def test_evidence_required(self):
        with pytest.raises(ValidationError):
            EvaluationOutput(
                session_id="s1",
                concept_coverage=0.8,
                logical_coherence=0.7,
                causal_reasoning=0.6,
                error_awareness=0.5,
                specificity=0.4,
                mastery_score=0.6,
                session_status="CONTINUE",
                guessing_detected=False,
            )

    def test_guessing_detected_true(self):
        out = _valid_output(guessing_detected=True)
        assert out.guessing_detected is True

    def test_guessing_detected_false(self):
        out = _valid_output(guessing_detected=False)
        assert out.guessing_detected is False

    def test_multiple_evidence_items_accepted(self):
        items = [
            EvidenceItem(dimension="concept_coverage", quote="osmosis is key"),
            EvidenceItem(dimension="logical_coherence", quote="the argument flows"),
            EvidenceItem(dimension="causal_reasoning", quote="because of X"),
            EvidenceItem(dimension="error_awareness", quote="no errors noted"),
            EvidenceItem(dimension="specificity", quote="very detailed answer here"),
        ]
        out = _valid_output(evidence=items)
        assert len(out.evidence) == 5

    def test_bloom_fields_still_optional(self):
        out = _valid_output(target_bloom_level="ANALYZE", demonstrated_bloom_level="REMEMBER")
        assert out.target_bloom_level == "ANALYZE"
        assert out.demonstrated_bloom_level == "REMEMBER"


# ---------------------------------------------------------------------------
# build_evidence tests
# ---------------------------------------------------------------------------

class TestBuildEvidence:
    def test_full_answer_produces_five_dimensions(self):
        items = build_evidence(_LONG_ANSWER, ["osmosis", "semipermeable", "concentration"])
        assert len(items) == 5
        dims = {e.dimension for e in items}
        assert dims == {
            "concept_coverage", "logical_coherence", "causal_reasoning",
            "error_awareness", "specificity",
        }

    def test_all_quotes_within_200_chars(self):
        items = build_evidence(_LONG_ANSWER, ["osmosis"])
        for item in items:
            assert len(item.quote) <= 200

    def test_empty_answer_yields_no_evidence(self):
        items = build_evidence("", ["osmosis"])
        assert items == []

    def test_short_answer_still_produces_evidence(self):
        items = build_evidence("Osmosis.", ["osmosis"])
        assert len(items) >= 1
        for item in items:
            assert 1 <= len(item.quote) <= 200

    def test_causal_marker_picks_causal_sentence(self):
        answer = (
            "Osmosis is the movement of water. Because the membrane is "
            "semipermeable, water flows toward higher concentration."
        )
        items = build_evidence(answer, ["osmosis"])
        causal = next(e for e in items if e.dimension == "causal_reasoning")
        assert "Because" in causal.quote or "because" in causal.quote

    def test_specificity_picks_longest_sentence(self):
        answer = (
            "Short sentence. A much longer and more detailed sentence "
            "about osmosis and semipermeable membranes that explains how "
            "this process works in living cells."
        )
        items = build_evidence(answer, ["osmosis"])
        spec = next(e for e in items if e.dimension == "specificity")
        assert "much longer" in spec.quote


# ---------------------------------------------------------------------------
# build_evaluation_output with evidence + guessing_detected
# ---------------------------------------------------------------------------

class TestBuildOutputWithEvidence:
    def test_output_has_evidence_and_guessing(self):
        out = build_evaluation_output(
            session_id="s1",
            analysis=_analysis(guessing_detected=True),
            mastery_score=0.7,
            session_status="CONTINUE",
            student_answer=_LONG_ANSWER,
            key_concepts=["osmosis", "semipermeable"],
            guessing_detected=True,
        )
        assert out.guessing_detected is True
        assert len(out.evidence) == 5
        assert all(isinstance(e, EvidenceItem) for e in out.evidence)

    def test_empty_answer_rejects(self):
        with pytest.raises(ValidationError):
            build_evaluation_output(
                session_id="s1",
                analysis=_analysis(),
                mastery_score=0.7,
                session_status="CONTINUE",
                student_answer="",
                key_concepts=["osmosis"],
            )

    def test_guessing_detected_defaults_from_analysis(self):
        out = build_evaluation_output(
            session_id="s1",
            analysis=_analysis(guessing_detected=True),
            mastery_score=0.7,
            session_status="CONTINUE",
            student_answer=_LONG_ANSWER,
            key_concepts=["osmosis"],
        )
        assert out.guessing_detected is True


# ---------------------------------------------------------------------------
# MasteryScorer smoothing guard unchanged
# ---------------------------------------------------------------------------

class TestMasteryScorerRetained:
    def test_smoothing_guard_caps_delta(self):
        scorer = MasteryScorer()
        analysis = _analysis()
        score = scorer.compute_mastery_score(
            analysis, concept_score=0.8, last_valid_score=0.5
        )
        assert 0.0 <= score <= 1.0
        assert abs(score - 0.5) <= 0.12 + 0.01  # max_step = 0.12

    def test_smoothing_guard_applies_with_llm_score(self):
        scorer = MasteryScorer()
        # answer_feedback with an extractable score
        analysis = _analysis(answer_feedback="Score: 0.9")
        score = scorer.compute_mastery_score(
            analysis, concept_score=0.8, last_valid_score=0.5
        )
        assert 0.0 <= score <= 1.0
        assert abs(score - 0.5) <= 0.12 + 0.01

    def test_mastery_without_llm_score_uses_concept_coverage(self):
        scorer = MasteryScorer()
        analysis = _analysis(answer_feedback="no score here")
        score = scorer.compute_mastery_score(
            analysis, concept_score=0.9, last_valid_score=0.5
        )
        assert 0.0 <= score <= 1.0
        assert abs(score - 0.5) <= 0.12 + 0.01


# ---------------------------------------------------------------------------
# Agent end-to-end: output carries evidence + guessing_detected
# ---------------------------------------------------------------------------

class TestAgentEvidenceIntegration:
    def test_handle_answer_output_has_evidence(self):
        from agents.evaluator.agent import EvaluatorAgent

        agent = EvaluatorAgent(require_llm=False)
        started = agent.start_session(
            task_title="Osmosis",
            task_description="Movement of water",
            task_details=(
                "Osmosis is the movement of water across a semipermeable membrane. "
                "Key concepts: osmosis, semipermeable, concentration gradient, plasmalemma."
            ),
        )
        result = agent.handle_user_answer(
            started["session_id"],
            "Osmosis is the movement of water across a semipermeable membrane "
            "because of a concentration gradient.",
        )
        out = result["evaluation_output"]
        assert "guessing_detected" in out
        assert isinstance(out["evidence"], list)
        assert len(out["evidence"]) >= 1
        for item in out["evidence"]:
            assert "dimension" in item and "quote" in item
            assert 1 <= len(item["quote"]) <= 200
