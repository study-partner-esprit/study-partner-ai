from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from agents.coach.agent import run_coach
from agents.coach.models.schemas import (
    CoachInput, ScheduledTask, FocusState
)


@patch('agents.coach.services.planner_repository.PlannerRepository.get_scheduled_tasks')
@patch('agents.coach.decision.llm_decider.call_gemini')
def test_coach_pipeline(mock_call_gemini, mock_get_tasks):
    # Mock scheduled tasks
    mock_tasks = [
        ScheduledTask(
            task_id="t1",
            title="MapReduce lecture",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=45),
            priority=1
        )
    ]
    mock_get_tasks.return_value = mock_tasks

    # Mock Gemini response
    mock_call_gemini.return_value = '{"action_type": "encourage", "message": "Keep going!", "reasoning": "User is focused", "target_task_id": "t1"}'

    data = CoachInput(
        scheduled_tasks=[],  # Will be overridden
        current_time=datetime.now(),
        focus_state=FocusState(state="Drifting", score=0.4),
        fatigue_probability=0.3,
        affective_state="engaged",
        ignored_count=0,
        do_not_disturb=False
    )

    action = run_coach(data)

    assert action.action_type == "encourage"
    assert action.message == "Keep going!"
    mock_get_tasks.assert_called_once()
    mock_call_gemini.assert_called_once()
