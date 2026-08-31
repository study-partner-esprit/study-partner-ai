"""Evaluation output schema + builder tests (F04 / EVAL-04).

Verifies:
- the five dimension scores are each bounded to [0.0, 1.0]
- out-of-range values are rejected
- unknown/extra fields are rejected (extra="forbid")
- the deterministic `specificity` builder yields an in-bounds score
- the agent attaches a validated EvaluationOutput to step results
"""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from agents.evaluator.schemas import (
    LLMAnalysisResponse,
    EvaluationOutput,
    build_evaluation_output,
)


def _analysis(**overrides) -> LLMAnalysisResponse:
    base = dict(
        concept_coverage=0.8,
        logical_coherence=0.7,
        causal_reasoning=0.6,
        error_awareness=0.5,
        answer_feedback="Good understanding of osmosis.",
        guessing_detected=False,
        missing_concepts=["plasmalemma"],
        misconceptions=[],
    )
    base.update(overrides)
    return LLMAnalysisResponse(**base)


class TestEvaluationOutputSchema:
    def test_valid_output_constructs(self):
        out = EvaluationOutput(
            session_id="s1",
            concept_coverage=0.8,
            logical_coherence=0.7,
            causal_reasoning=0.6,
            error_awareness=0.5,
            specificity=0.9,
            mastery_score=0.75,
            next_question="Why does this matter?",
            session_status="CONTINUE",
        )
        assert out.mastery_score == 0.75
        assert out.next_question == "Why does this matter?"

    def test_all_five_dimensions_present(self):
        out = EvaluationOutput(
            session_id="s1",
            concept_coverage=0.8,
            logical_coherence=0.7,
            causal_reasoning=0.6,
            error_awareness=0.5,
            specificity=0.4,
            mastery_score=0.6,
            session_status="CONTINUE",
        )
        dims = [
            out.concept_coverage,
            out.logical_coherence,
            out.causal_reasoning,
            out.error_awareness,
            out.specificity,
        ]
        assert len(dims) == 5
        assert all(0.0 <= d <= 1.0 for d in dims)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("concept_coverage", 1.5),
            ("logical_coherence", -0.1),
            ("causal_reasoning", 2.0),
            ("error_awareness", 1.01),
            ("specificity", -0.5),
            ("mastery_score", 1.2),
        ],
    )
    def test_out_of_range_rejected(self, field, value):
        kwargs = dict(
            session_id="s1",
            concept_coverage=0.8,
            logical_coherence=0.7,
            causal_reasoning=0.6,
            error_awareness=0.5,
            specificity=0.4,
            mastery_score=0.6,
            session_status="CONTINUE",
        )
        kwargs[field] = value
        with pytest.raises(ValidationError):
            EvaluationOutput(**kwargs)

    def test_unknown_extra_field_rejected(self):
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
                hacks="give me 1.0",
            )

    def test_bloom_fields_null_by_default_and_settable(self):
        out = EvaluationOutput(
            session_id="s1",
            concept_coverage=0.8,
            logical_coherence=0.7,
            causal_reasoning=0.6,
            error_awareness=0.5,
            specificity=0.4,
            mastery_score=0.6,
            session_status="CONTINUE",
        )
        assert out.target_bloom_level is None
        assert out.demonstrated_bloom_level is None

        out2 = EvaluationOutput(
            session_id="s1",
            concept_coverage=0.8,
            logical_coherence=0.7,
            causal_reasoning=0.6,
            error_awareness=0.5,
            specificity=0.4,
            mastery_score=0.6,
            session_status="CONTINUE",
            target_bloom_level="ANALYZE",
            demonstrated_bloom_level="REMEMBER",
        )
        assert out2.target_bloom_level == "ANALYZE"
        assert out2.demonstrated_bloom_level == "REMEMBER"
        assert out2.target_bloom_level != out2.demonstrated_bloom_level


class TestBuildEvaluationOutput:
    def test_builds_valid_output_from_analysis(self):
        out = build_evaluation_output(
            session_id="sess-1",
            analysis=_analysis(),
            mastery_score=0.73,
            session_status="CONTINUE",
            student_answer="Water moves across a semipermeable membrane by osmosis toward the higher solute concentration.",
            key_concepts=["osmosis", "semipermeable", "plasmalemma"],
            next_question="How does this relate to cell turgor?",
        )
        assert isinstance(out, EvaluationOutput)
        assert out.session_status == "CONTINUE"
        assert out.next_question == "How does this relate to cell turgor?"
        assert 0.0 <= out.specificity <= 1.0

    def test_empty_answer_specificity_zero(self):
        out = build_evaluation_output(
            session_id="s1",
            analysis=_analysis(),
            mastery_score=0.5,
            session_status="CONTINUE",
            student_answer="",
            key_concepts=["osmosis"],
        )
        assert out.specificity == 0.0

    def test_specific_key_answer_scores_high(self):
        out = build_evaluation_output(
            session_id="s1",
            analysis=_analysis(),
            mastery_score=0.5,
            session_status="CONTINUE",
            student_answer="Osmosis is the movement of water across a semipermeable membrane driven by concentration gradient differences.",
            key_concepts=["osmosis", "semipermeable", "concentration"],
        )
        assert out.specificity >= 0.4

    def test_bloom_fields_passthrough(self):
        out = build_evaluation_output(
            session_id="s1",
            analysis=_analysis(),
            mastery_score=0.5,
            session_status="CONTINUE",
            student_answer="ok",
            target_bloom_level="ANALYZE",
            demonstrated_bloom_level="REMEMBER",
        )
        assert out.target_bloom_level == "ANALYZE"
        assert out.demonstrated_bloom_level == "REMEMBER"


class TestAgentAttachesOutput:
    def test_handle_answer_returns_structured_output(self):
        from agents.evaluator.agent import EvaluatorAgent

        agent = EvaluatorAgent(require_llm=False)
        started = agent.start_session(
            task_title="Osmosis",
            task_description="Movement of water",
            task_details="Osmosis is the movement of water across a semipermeable membrane toward higher solute concentration. Key concepts: osmosis, semipermeable, concentration gradient, plasmalemma.",
        )
        result = agent.handle_user_answer(
            started["session_id"],
            "Water moves across the membrane by osmosis toward higher concentration.",
        )
        assert "evaluation_output" in result
        out = result["evaluation_output"]
        assert out["session_id"] == started["session_id"]
        assert 0.0 <= out["mastery_score"] <= 1.0
        assert out["session_status"] in {"CONTINUE", "MASTERY_CONFIRMED", "FAILED"}
        for dim in ("concept_coverage", "logical_coherence", "causal_reasoning",
                    "error_awareness", "specificity"):
            assert 0.0 <= out[dim] <= 1.0
