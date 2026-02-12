from pydantic import BaseModel
from typing import List, Optional, Literal, TYPE_CHECKING, Any
from datetime import datetime

if TYPE_CHECKING:
    from services.signal_processing_service.signal_snapshot import SignalSnapshot


class ScheduledTask(BaseModel):
    task_id: str
    title: str
    start_time: datetime
    end_time: datetime
    priority: int


class FocusState(BaseModel):
    state: Literal["Focused", "Drifting", "Lost"]
    score: float


class FatigueState(BaseModel):
    state: Literal["Alert", "Moderate", "High", "Critical"]
    score: float


class ScheduleChange(BaseModel):
    """Represents a scheduling change requested by the Coach agent."""
    action: Literal["add_break", "extend_task", "reschedule_task", "cancel_task", "suspend_session"]
    duration_minutes: Optional[int] = None
    new_start_time: Optional[datetime] = None
    affected_task_ids: List[str] = []
    reasoning: str = ""


class CoachInput(BaseModel):
    scheduled_tasks: List[ScheduledTask]
    current_time: datetime
    focus_state: FocusState
    fatigue_state: FatigueState
    affective_state: Literal[
        "engaged", "frustrated", "stressed", "bored", "confident"
    ]
    ignored_count: int = 0
    do_not_disturb: bool = False
    is_late: bool = False
    signals: Optional[Any] = None  # SignalSnapshot from ML models


class CoachAction(BaseModel):
    action_type: Literal[
        "nudge",
        "encourage",
        "suggest_break",
        "renegotiate_task",
        "silence"
    ]
    message: Optional[str] = None
    reasoning: str
    target_task_id: Optional[str] = None
    
    # NEW: Optional scheduling directives for autonomous execution
    schedule_changes: Optional[ScheduleChange] = None
