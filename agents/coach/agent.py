from agents.coach.models.schemas import CoachInput, CoachAction
from agents.coach.rules.rule_engine import apply_rules
from agents.coach.decision.llm_decider import decide_with_llm
from agents.coach.services.planner_repository import PlannerRepository
from agents.coach.services.coach_history_repository import CoachHistoryRepository
from utils.logger import get_logger

logger = get_logger(__name__)
_history_repo = CoachHistoryRepository()


def run_coach(
    input_data: CoachInput,
    user_id: str = "",
    trace_id: str = "",
) -> CoachAction:
    """
    Main entry point for the Coach agent.

    Args:
        input_data:     Current student state.
        user_id:        User identifier — used for history lookup + persistence.
        trace_id:       Request-level trace ID for log correlation.

    Returns:
        CoachAction describing the intervention to apply.
    """
    logger.info(
        "coach_run_start",
        extra={
            "user_id": user_id,
            "trace_id": trace_id,
            "focus": input_data.focus_state.state,
            "fatigue": input_data.fatigue_state.state,
        },
    )

    # Fetch scheduled tasks from MongoDB
    repo = PlannerRepository()
    scheduled_tasks = repo.get_scheduled_tasks()
    input_data.scheduled_tasks = scheduled_tasks

    # Load recent coaching history so the LLM avoids repetitive interventions
    recent_history = _history_repo.get_recent_actions(user_id, limit=5)

    # Step 1: apply hard rules (may short-circuit LLM)
    rule_action = apply_rules(input_data)

    if rule_action is not None:
        logger.info(
            "coach_rule_fired",
            extra={
                "action_type": rule_action.action_type,
                "trace_id": trace_id,
            },
        )
        _persist_action(rule_action, input_data, user_id, trace_id)
        return rule_action

    # Step 2: LLM decides
    action = decide_with_llm(
        input_data,
        recent_history=recent_history,
        trace_id=trace_id,
    )
    _persist_action(action, input_data, user_id, trace_id)
    return action


def _persist_action(
    action: CoachAction,
    input_data: CoachInput,
    user_id: str,
    trace_id: str,
) -> None:
    """Fire-and-forget persistence — errors are logged, not raised."""
    if not user_id:
        return
    try:
        _history_repo.save_action(user_id, action, input_data, trace_id)
    except Exception as exc:
        logger.warning(
            "coach_persist_action_error",
            extra={"error": str(exc), "trace_id": trace_id},
        )

