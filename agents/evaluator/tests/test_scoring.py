"""MasteryScorer and prompt-helper unit tests."""

from __future__ import annotations

from agents.evaluator.prompts import clean_concepts, concept_coverage, extract_score
from agents.evaluator.schemas import LLMAnalysisResponse
from agents.evaluator.scoring import MasteryScorer


def make_analysis(feedback: str, concept_coverage: float = 0.5) -> LLMAnalysisResponse:
    return LLMAnalysisResponse(
        concept_coverage=concept_coverage,
        logical_coherence=0.5,
        causal_reasoning=0.5,
        error_awareness=0.5,
        answer_feedback=feedback,
        guessing_detected=False,
        missing_concepts=[],
        misconceptions=[],
    )


def test_extract_score_parses_valid_score():
    assert extract_score("Score: 0.85\nStrengths: Good") == 0.85


def test_extract_score_handles_missing():
    assert extract_score("No score here, just feedback") in (None, 0.0)


def test_score_bounded():
    s = MasteryScorer.compute_mastery_score(make_analysis("Score: 1.0"), concept_score=0.9)
    assert 0.0 <= s <= 1.0


def test_determine_state_mastery_requires_no_guessing():
    assert MasteryScorer.determine_state(0.9, guessing_detected=False, generic_answer=False) == "MASTERY_CONFIRMED"
    assert MasteryScorer.determine_state(0.9, guessing_detected=True, generic_answer=False) == "CONTINUE"


def test_determine_state_failure_below_threshold():
    assert MasteryScorer.determine_state(0.4, guessing_detected=False, generic_answer=False) == "FAILED"


def test_clean_concepts_filters_generic_words():
    cleaned = clean_concepts(["run", "machine learning", "jump", "data", "neural network"])
    assert "machine learning" in cleaned
    assert "neural network" in cleaned
    assert "run" not in cleaned


def test_concept_coverage_counts_matches():
    cov = concept_coverage(
        "neural networks and machine learning help train models",
        ["neural network", "machine learning", "gradient descent"],
    )
    assert cov > 0.4


def test_generate_missing_concept_feedback_empty_at_high_score():
    assert MasteryScorer.generate_missing_concept_feedback(
        0.8, ["neural network"], ["neural network"], threshold=0.7
    ) == ""
