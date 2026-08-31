from pydantic import BaseModel, ConfigDict, Field, field_validator
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


class SessionStats(BaseModel):
    """Bounded live session statistics (COACH-13).

    Acme bounds keep the trusted state block small and integer-clean:
    - progress_pct:       0–100 % of the session's tasks completed
    - minutes_elapsed:    0–600 minutes on task (0 when stale/missing)
    - task_switches:      0–50 task transitions so far
    - break_count:        0–20 breaks taken
    - current_streak_days:0–365 day study streak

    Missing/stale values default to 0 — they must never fail the job.
    """

    progress_pct: int = Field(default=0, ge=0, le=100)
    minutes_elapsed: int = Field(default=0, ge=0, le=600)
    task_switches: int = Field(default=0, ge=0, le=50)
    break_count: int = Field(default=0, ge=0, le=20)
    current_streak_days: int = Field(default=0, ge=0, le=365)


class ScheduleChange(BaseModel):
    """Represents a scheduling change requested by the Coach agent."""

    action: Literal[
        "add_break", "extend_task", "reschedule_task", "cancel_task", "suspend_session"
    ]
    duration_minutes: Optional[int] = None
    new_start_time: Optional[datetime] = None
    affected_task_ids: List[str] = []
    reasoning: str = ""


class CourseContext(BaseModel):
    """One bounded course-catalog entry (F03 / COACH-14).

    The coach loads the student's NEWEST enrolled courses (≤ 10) from the
    course catalog, each reduced to its subject title + course title + a
    bounded set of key concepts (≤ 15, each ≤ 100 chars). No files, URLs,
    descriptions, ids or any other field leaves the repository — nothing
    beyond these three strings can reach the prompt. All three are
    user-supplied content, so prompt.py wraps them inside UNTRUSTED DATA
    blocks (COACH-03/12) before the LLM ever sees them.
    """

    subject: str = Field(default="Unknown", max_length=60)
    title: str = Field(default="", max_length=100)
    key_concepts: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("key_concepts")
    @classmethod
    def _bounded_concepts(cls, v: List[str]) -> List[str]:
        # Defensive reducer: never fails the job, silently trims to the bound.
        cleaned = []
        for c in v:
            if isinstance(c, str) and c.strip():
                cleaned.append(c.strip()[:100])
        return cleaned[:15]


class CoachInput(BaseModel):
    scheduled_tasks: List[ScheduledTask]
    current_time: datetime
    focus_state: FocusState
    fatigue_state: FatigueState
    affective_state: Literal["engaged", "frustrated", "stressed", "bored", "confident"]
    ignored_count: int = 0
    do_not_disturb: bool = False
    is_late: bool = False
    signals: Optional[Any] = None  # SignalSnapshot from ML models
    session_stats: Optional[SessionStats] = None  # COACH-13 live session stats
    # COACH-14 bounded course catalog (newest ≥10; every field untrusted).
    catalog_courses: Optional[List[CourseContext]] = None

    # Current task context — enriches prompt and history logging
    current_task_title: Optional[str] = None
    current_task_difficulty: Optional[float] = None  # 0.0 – 1.0
    current_task_subject: Optional[str] = None
    current_task_key_concepts: Optional[List[str]] = None


class CoachOutput(BaseModel):
    """Strict user-facing coach output (F03 / COACH-05).

    The UI must only ever trust this validated structure — never free-form
    LLM text. Fields are constrained so malformed nudges are rejected early:
    - `nudge_text`: 1–500 chars
    - `intensity`:   0.0–1.0
    - `category`:    motivation | focus | fatigue | break
    """

    nudge_text: str = Field(min_length=1, max_length=500)
    intensity: float = Field(ge=0.0, le=1.0)
    category: Literal["motivation", "focus", "fatigue", "break"]


class CoachAction(BaseModel):
    action_type: Literal[
        "nudge", "encourage", "suggest_break", "renegotiate_task", "silence"
    ]
    message: Optional[str] = None
    reasoning: str
    target_task_id: Optional[str] = None

    # NEW: Optional scheduling directives for autonomous execution
    schedule_changes: Optional[ScheduleChange] = None

    # COACH-05: strict validated user-facing output (LLM path only) and a
    # sanitized error string when the LLM output cannot be parsed/validated.
    nudge: Optional[CoachOutput] = None
    coach_error: Optional[str] = None
