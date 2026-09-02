"""
Evaluation state machine.
Manages state transitions for the evaluation pipeline.
"""

from agents.evaluator.schemas import EvaluationState


class StateMachine:
    """Manages evaluation state transitions."""

    # Valid state transitions
    TRANSITIONS = {
        EvaluationState.CONTINUE: [
            EvaluationState.CONTINUE,
            EvaluationState.MASTERY_CONFIRMED,
            EvaluationState.FAILED,
        ],
        EvaluationState.MASTERY_CONFIRMED: [],
        EvaluationState.FAILED: [],
    }

    @staticmethod
    def validate_transition(
        current: EvaluationState,
        next_state: EvaluationState,
    ) -> bool:
        """
        Validate if transition is allowed.

        Args:
            current: Current state
            next_state: Desired next state

        Returns:
            True if transition is valid
        """
        allowed = StateMachine.TRANSITIONS.get(current, [])
        return next_state in allowed

    @staticmethod
    def next_state(
        current: EvaluationState,
        mastery_score: float,
    ) -> EvaluationState:
        """
        Determine next state based on mastery score.

        Args:
            current: Current state
            mastery_score: Mastery score (0.0-1.0)

        Returns:
            Next state based on scoring thresholds
        """
        if mastery_score >= 0.85:
            return EvaluationState.MASTERY_CONFIRMED
        elif mastery_score < 0.60:
            return EvaluationState.FAILED
        else:
            return EvaluationState.CONTINUE
