from typing import Optional
from agents.coach.models.schemas import CoachInput, CoachAction


def apply_rules(input_data: CoachInput) -> Optional[CoachAction]:

    if input_data.do_not_disturb:
        return CoachAction(
            action_type="silence",
            message=None,
            reasoning="Do not disturb is enabled."
        )

    if input_data.ignored_count >= 3:
        return CoachAction(
            action_type="silence",
            message=None,
            reasoning="Coach was ignored several times."
        )

    if input_data.focus_state.state == "Focused":
        return CoachAction(
            action_type="silence",
            message=None,
            reasoning="User is deeply focused."
        )

    return None
