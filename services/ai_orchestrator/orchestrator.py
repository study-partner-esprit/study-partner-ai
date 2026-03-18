"""AI Orchestrator service for coordinating Coach agent execution.

This service integrates signal processing with the Coach agent,
fetching both scheduled tasks and ML signals before running the coach.
Parallel I/O via ThreadPoolExecutor; trace_id propagation throughout.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from datetime import datetime

from agents.coach.agent import run_coach
from agents.coach.models.schemas import (
    CoachInput,
    CoachAction,
    ScheduledTask,
    FocusState,
    FatigueState,
)
from agents.coach.services.planner_repository import PlannerRepository
from services.signal_processing_service.service import SignalProcessingService
from services.signal_processing_service.signal_snapshot import SignalSnapshot
from utils.logger import get_logger

logger = get_logger(__name__)


class AIOrchestrator:
    """
    Orchestrates the execution of the Coach agent with full context.

    This service:
    1. Fetches scheduled tasks and ML signals in parallel (ThreadPoolExecutor)
    2. Constructs the CoachInput with all necessary data
    3. Executes the Coach agent (with trace_id propagation)
    4. Returns the Coach's decision/action
    """

    def __init__(self):
        """Initialize the orchestrator with required services."""
        self.signal_service = SignalProcessingService()
        self.planner_repo = PlannerRepository()

    def run_coach(
        self,
        user_id: str,
        current_time: Optional[datetime] = None,
        ignored_count: int = 0,
        do_not_disturb: bool = False,
        trace_id: Optional[str] = None,
        live_focus_score: Optional[float] = None,
        live_focus_state: Optional[str] = None,
        live_fatigue_score: Optional[float] = None,
        live_fatigue_state: Optional[str] = None,
    ) -> CoachAction:
        """
        Execute the Coach agent with full user context.

        Args:
            user_id:        The user's unique identifier.
            current_time:   The current time (defaults to now).
            ignored_count:  Number of times user has ignored recent nudges.
            do_not_disturb: Whether user has enabled DND mode.
            trace_id:       Optional request trace ID; generated if not provided.

        Returns:
            A CoachAction containing the coach's decision.
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        if current_time is None:
            current_time = datetime.now()

        logger.info(
            "orchestrator_run_coach_start",
            extra={"user_id": user_id, "trace_id": trace_id},
        )

        # --- Check if live signals were provided by the frontend --- #
        has_live_signals = (
            live_focus_score is not None
            or live_focus_state is not None
            or live_fatigue_score is not None
            or live_fatigue_state is not None
        )

        # --- Parallel I/O: fetch tasks + signal snapshot simultaneously --- #
        scheduled_tasks: list[ScheduledTask] = []
        signal_snapshot: Optional[SignalSnapshot] = None

        if has_live_signals:
            # Skip expensive DB lookup — use live webcam values directly
            logger.info(
                "orchestrator_using_live_signals",
                extra={"user_id": user_id, "trace_id": trace_id},
            )
            # Only fetch scheduled tasks
            scheduled_tasks = self._fetch_scheduled_tasks()
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_tasks = pool.submit(self._fetch_scheduled_tasks)
                fut_signals = pool.submit(self._fetch_signal_snapshot, user_id)

                for fut in as_completed([fut_tasks, fut_signals]):
                    try:
                        result = fut.result()
                        if fut is fut_tasks:
                            scheduled_tasks = result
                        else:
                            signal_snapshot = result
                    except Exception as exc:
                        logger.warning(
                            "orchestrator_parallel_fetch_error",
                            extra={"trace_id": trace_id, "error": str(exc)},
                        )

        # Step 3: Build CoachInput
        coach_input = self._build_coach_input(
            user_id=user_id,
            scheduled_tasks=scheduled_tasks,
            signal_snapshot=signal_snapshot,
            current_time=current_time,
            ignored_count=ignored_count,
            do_not_disturb=do_not_disturb,
            live_focus_score=live_focus_score,
            live_focus_state=live_focus_state,
            live_fatigue_score=live_fatigue_score,
            live_fatigue_state=live_fatigue_state,
        )

        # Step 4: Execute Coach agent (history lookup + action persistence inside)
        coach_action = run_coach(
            coach_input,
            user_id=user_id,
            trace_id=trace_id,
        )

        logger.info(
            "orchestrator_run_coach_done",
            extra={
                "user_id": user_id,
                "trace_id": trace_id,
                "action_type": coach_action.action_type,
            },
        )
        return coach_action

    def _fetch_scheduled_tasks(self) -> list[ScheduledTask]:
        """
        Fetch scheduled tasks for the user.

        Returns:
            List of scheduled tasks (may be empty)
        """
        try:
            tasks = self.planner_repo.get_scheduled_tasks()
            return tasks if tasks else []
        except Exception as e:
            logger.warning("orchestrator_fetch_tasks_error", extra={"error": str(e)})
            return []

    def _fetch_signal_snapshot(self, user_id: str) -> Optional[SignalSnapshot]:
        """
        Fetch the latest signal snapshot for the user.

        Returns:
            SignalSnapshot if available, None otherwise.
        """
        try:
            snapshot = self.signal_service.get_latest_snapshot(user_id)

            if snapshot is None:
                logger.info(
                    "orchestrator_generate_new_snapshot",
                    extra={"user_id": user_id},
                )
                snapshot = self.signal_service.get_current_signal_snapshot(user_id)
            else:
                age_seconds = (datetime.now() - snapshot.timestamp).total_seconds()
                if age_seconds > 120:
                    logger.info(
                        "orchestrator_snapshot_stale",
                        extra={"user_id": user_id, "age_seconds": int(age_seconds)},
                    )
                    snapshot = self.signal_service.get_current_signal_snapshot(user_id)

            return snapshot
        except Exception as e:
            logger.warning(
                "orchestrator_fetch_snapshot_error",
                extra={"user_id": user_id, "error": str(e)},
            )
            return None

    def _build_coach_input(
        self,
        user_id: str,
        scheduled_tasks: list[ScheduledTask],
        signal_snapshot: Optional[SignalSnapshot],
        current_time: datetime,
        ignored_count: int,
        do_not_disturb: bool,
        live_focus_score: Optional[float] = None,
        live_focus_state: Optional[str] = None,
        live_fatigue_score: Optional[float] = None,
        live_fatigue_state: Optional[str] = None,
    ) -> CoachInput:
        """
        Build the CoachInput from all available data.
        Live signal params take precedence over DB snapshot.

        Returns:
            A fully populated CoachInput.
        """
        # Priority: live webcam signals > DB snapshot > defaults
        if live_focus_score is not None or live_focus_state is not None:
            focus_state = FocusState(
                state=live_focus_state or "Drifting",
                score=live_focus_score if live_focus_score is not None else 0.5,
            )
        elif signal_snapshot is not None:
            focus_state = FocusState(
                state=signal_snapshot.focus_state,
                score=signal_snapshot.focus_score,
            )
        else:
            focus_state = FocusState(state="Drifting", score=0.5)

        if live_fatigue_score is not None or live_fatigue_state is not None:
            fatigue_state = FatigueState(
                state=live_fatigue_state or "Moderate",
                score=live_fatigue_score if live_fatigue_score is not None else 0.3,
            )
        elif signal_snapshot is not None:
            fatigue_state = FatigueState(
                state=signal_snapshot.fatigue_state,
                score=signal_snapshot.fatigue_score,
            )
        else:
            fatigue_state = FatigueState(state="Moderate", score=0.3)

        affective_state = "engaged"  # derive from signals in future
        is_late = self._check_if_late(scheduled_tasks, current_time)

        return CoachInput(
            scheduled_tasks=scheduled_tasks,
            current_time=current_time,
            focus_state=focus_state,
            fatigue_state=fatigue_state,
            affective_state=affective_state,
            ignored_count=ignored_count,
            do_not_disturb=do_not_disturb,
            is_late=is_late,
            signals=signal_snapshot,
        )

    def _check_if_late(
        self,
        scheduled_tasks: list[ScheduledTask],
        current_time: datetime,
    ) -> bool:
        """Return True if the user is late for any scheduled task."""
        for task in scheduled_tasks:
            if current_time > task.start_time:
                return True
        return False
