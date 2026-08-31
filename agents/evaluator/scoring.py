"""
Deterministic mastery scoring using pure Python logic.
"""

import logging
from agents.evaluator.schemas import LLMAnalysisResponse
from agents.evaluator.prompts import extract_score

logger = logging.getLogger(__name__)


class MasteryScorer:
    """Deterministic mastery score computation."""

    # Thresholds
    MASTERY_THRESHOLD = 0.85
    FAILURE_THRESHOLD = 0.60

    @staticmethod
    def compute_mastery_score(analysis: LLMAnalysisResponse, concept_score: float = 0.5, last_valid_score: float = 0.5) -> float:
        """
        Compute mastery score using hybrid formula with robust fallback.
        Combines LLM score with local concept coverage analysis.
        If LLM parsing fails, uses local concept coverage as primary score.

        Formula when LLM score available:
        mastery = 0.6 * llm_score + 0.4 * concept_coverage

        Formula when LLM parsing fails:
        mastery = 0.4 * concept_coverage + 0.6 * last_valid_score

        Args:
            analysis: LLM analysis result (may have partial fields)
            concept_score: Local concept coverage score (0.0-1.0), default 0.5 if not computed
            last_valid_score: Last valid score from previous attempts (0.0-1.0), default 0.5

        Returns:
            Mastery score (0.0-1.0) with percentage conversion for display
        """
        # Extract LLM score from feedback text
        llm_score = None
        if hasattr(analysis, 'answer_feedback') and analysis.answer_feedback:
            # Try to extract score from feedback text using improved parsing
            llm_score = extract_score(analysis.answer_feedback)

        # Get concept coverage score
        concept_coverage = getattr(analysis, 'concept_coverage', concept_score)
        if concept_coverage is None:
            concept_coverage = concept_score

        # Ensure valid ranges
        if concept_coverage is not None:
            concept_coverage = max(0.0, min(1.0, float(concept_coverage)))
        else:
            concept_coverage = 0.5

        last_valid_score = max(0.0, min(1.0, float(last_valid_score)))

        # Smooth and cap changes to reduce score drift between attempts
        max_step = 0.12
        if llm_score is not None:
            llm_score = max(0.0, min(1.0, float(llm_score)))
            raw_target = 0.6 * llm_score + 0.4 * concept_coverage
            raw_target = max(0.0, min(1.0, raw_target))
            mastery = last_valid_score + 0.4 * (raw_target - last_valid_score)
            delta = mastery - last_valid_score
            delta = max(min(delta, max_step), -max_step)
            mastery = last_valid_score + delta
            logger.debug(
                f"Smoothed mastery from {last_valid_score:.3f} toward {raw_target:.3f} => {mastery:.3f} (delta {delta:.3f})"
            )
        else:
            raw_target = 0.4 * concept_coverage + 0.6 * last_valid_score
            mastery = last_valid_score + 0.4 * (raw_target - last_valid_score)
            delta = mastery - last_valid_score
            delta = max(min(delta, max_step), -max_step)
            mastery = last_valid_score + delta
            logger.warning(
                f"LLM score parsing failed, smoothing fallback: from {last_valid_score:.3f} toward {raw_target:.3f} => {mastery:.3f}"
            )

        # Clamp final result and round
        mastery = max(0.0, min(1.0, mastery))
        mastery = round(mastery, 3)

        # Log both scores for debugging
        mastery_percentage = int(mastery * 100)
        logger.info(f"Mastery score: {mastery} ({mastery_percentage}%) - LLM: {llm_score}, Concept: {concept_coverage}, Last Valid: {last_valid_score}")

        return mastery

    @staticmethod
    def determine_state(
        mastery_score: float,
        guessing_detected: bool,
        generic_answer: bool,
    ) -> str:
        """
        Determine evaluation state based on deterministic rules.

        Args:
            mastery_score: Computed mastery score
            guessing_detected: Whether LLM detected guessing
            generic_answer: Whether answer is too generic

        Returns:
            State: "MASTERY_CONFIRMED", "FAILED", or "CONTINUE"
        """
        # Mastery requires high score AND no guessing/generic answers
        if mastery_score >= MasteryScorer.MASTERY_THRESHOLD:
            if not (guessing_detected or generic_answer):
                return "MASTERY_CONFIRMED"

        # Failure state
        if mastery_score < MasteryScorer.FAILURE_THRESHOLD:
            return "FAILED"

        # Partial - continue evaluation
        return "CONTINUE"

    @staticmethod
    def generate_missing_concept_feedback(
        mastery_score: float,
        missing_concepts: list[str],
        key_concepts: list[str],
        threshold: float = 0.7
    ) -> str:
        """
        Generate specific, actionable feedback about missing concepts.
        Only includes relevant scientific/domain concepts, max 5.

        Args:
            mastery_score: Current mastery score
            missing_concepts: List of concepts identified as missing
            key_concepts: Full list of key concepts for the task
            threshold: Score threshold below which to provide detailed feedback

        Returns:
            Specific feedback string, or empty string if score >= threshold
        """
        if mastery_score >= threshold or not missing_concepts:
            return ""

        # Clean missing concepts to ensure they're meaningful
        from agents.evaluator.prompts import clean_concepts
        cleaned_missing = clean_concepts(missing_concepts)

        # Limit to top 5 most critical missing concepts
        critical_missing = cleaned_missing[:5]

        if not critical_missing:
            return ""

        feedback_parts = []

        # Build specific feedback based on what's missing
        if len(critical_missing) == 1:
            feedback_parts.append(f"You might want to focus on understanding '{critical_missing[0]}' better.")
        elif len(critical_missing) <= 3:
            concepts_str = "', '".join(critical_missing)
            feedback_parts.append(f"Consider reviewing these key concepts: '{concepts_str}'.")
        else:
            # Show top 3, mention others exist
            top_three = critical_missing[:3]
            concepts_str = "', '".join(top_three)
            feedback_parts.append(f"Key areas to focus on: '{concepts_str}'.")
            remaining = len(critical_missing) - 3
            if remaining > 0:
                feedback_parts.append(f"Plus {remaining} other important concepts.")

        # Add actionable guidance
        feedback_parts.append("Try explaining these concepts in your next response with specific examples.")

        return " ".join(feedback_parts)
