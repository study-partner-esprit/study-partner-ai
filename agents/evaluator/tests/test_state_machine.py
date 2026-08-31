"""State machine transition tests (EVAL-01: multi-turn state retained, not simplified)."""

from __future__ import annotations

from agents.evaluator.schemas import EvaluationState
from agents.evaluator.state_machine import StateMachine


def test_continue_allows_all_progressions():
    for next_state in (
        EvaluationState.CONTINUE,
        EvaluationState.MASTERY_CONFIRMED,
        EvaluationState.FAILED,
    ):
        assert StateMachine.validate_transition(EvaluationState.CONTINUE, next_state)


def test_terminal_states_are_sinks():
    for terminal in (EvaluationState.MASTERY_CONFIRMED, EvaluationState.FAILED):
        assert not StateMachine.validate_transition(terminal, EvaluationState.CONTINUE)
        assert not StateMachine.validate_transition(terminal, EvaluationState.MASTERY_CONFIRMED)


def test_next_state_by_mastery_thresholds():
    assert StateMachine.next_state(EvaluationState.CONTINUE, 0.95) == EvaluationState.MASTERY_CONFIRMED
    assert StateMachine.next_state(EvaluationState.CONTINUE, 0.50) == EvaluationState.FAILED
    assert StateMachine.next_state(EvaluationState.CONTINUE, 0.70) == EvaluationState.CONTINUE
