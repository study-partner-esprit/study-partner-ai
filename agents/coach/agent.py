from agents.coach.models.schemas import CoachInput, CoachAction
from agents.coach.rules.rule_engine import apply_rules
from agents.coach.decision.llm_decider import decide_with_llm
from agents.coach.services.planner_repository import PlannerRepository


def run_coach(input_data: CoachInput) -> CoachAction:
    # Fetch scheduled tasks from MongoDB
    repo = PlannerRepository()
    scheduled_tasks = repo.get_scheduled_tasks()
    input_data.scheduled_tasks = scheduled_tasks

    rule_action = apply_rules(input_data)

    if rule_action is not None:
        return rule_action

    return decide_with_llm(input_data)
