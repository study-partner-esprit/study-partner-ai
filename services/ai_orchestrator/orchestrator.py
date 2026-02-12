"""AI Orchestrator service for coordinating Coach agent execution.

This service integrates signal processing with the Coach agent,
fetching both scheduled tasks and ML signals before running the coach.
"""

from typing import Optional, Any
from datetime import datetime

from agents.coach.agent import run_coach
from agents.coach.models.schemas import CoachInput, CoachAction, ScheduledTask, FocusState, FatigueState
from agents.coach.services.planner_repository import PlannerRepository
from services.signal_processing_service.service import SignalProcessingService
from services.signal_processing_service.signal_snapshot import SignalSnapshot


class AIOrchestrator:
    """
    Orchestrates the execution of the Coach agent with full context.
    
    This service:
    1. Fetches scheduled tasks from MongoDB
    2. Fetches the latest signal snapshot (or creates one)
    3. Constructs the CoachInput with all necessary data
    4. Executes the Coach agent
    5. Returns the Coach's decision/action
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
        do_not_disturb: bool = False
    ) -> CoachAction:
        """
        Execute the Coach agent with full user context.
        
        Args:
            user_id: The user's unique identifier
            current_time: The current time (defaults to now)
            ignored_count: Number of times user has ignored recent nudges
            do_not_disturb: Whether user has enabled DND mode
        
        Returns:
            A CoachAction containing the coach's decision
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Step 1: Fetch scheduled tasks from MongoDB
        print(f"Fetching scheduled tasks for user {user_id}...")
        scheduled_tasks = self._fetch_scheduled_tasks()
        
        # Step 2: Fetch or generate signal snapshot
        print(f"Fetching signal snapshot for user {user_id}...")
        signal_snapshot = self._fetch_signal_snapshot(user_id)
        
        # Step 3: Build CoachInput
        coach_input = self._build_coach_input(
            user_id=user_id,
            scheduled_tasks=scheduled_tasks,
            signal_snapshot=signal_snapshot,
            current_time=current_time,
            ignored_count=ignored_count,
            do_not_disturb=do_not_disturb
        )
        
        # Step 4: Execute Coach agent
        print("Executing Coach agent...")
        coach_action = run_coach(coach_input)
        
        print(f"Coach decision: {coach_action.action_type}")
        return coach_action
    
    def _fetch_scheduled_tasks(self) -> list[ScheduledTask]:
        """
        Fetch scheduled tasks for the user.
        
        Args:
            user_id: The user's unique identifier
        
        Returns:
            List of scheduled tasks (may be empty)
        """
        try:
            tasks = self.planner_repo.get_scheduled_tasks()
            return tasks if tasks else []
        except Exception as e:
            print(f"Warning: Could not fetch scheduled tasks: {e}")
            return []
    
    def _fetch_signal_snapshot(self, user_id: str) -> Optional[SignalSnapshot]:
        """
        Fetch the latest signal snapshot for the user.
        
        Args:
            user_id: The user's unique identifier
        
        Returns:
            SignalSnapshot if available, None otherwise
        """
        try:
            # Try to get the latest snapshot
            snapshot = self.signal_service.get_latest_snapshot(user_id)
            
            # If no snapshot exists or it's too old, generate a new one
            if snapshot is None:
                print("No signal snapshot found, generating new one...")
                snapshot = self.signal_service.get_current_signal_snapshot(user_id)
            else:
                # Check if snapshot is recent (within 2 minutes)
                age_seconds = (datetime.now() - snapshot.timestamp).total_seconds()
                if age_seconds > 120:  # 2 minutes
                    print(f"Signal snapshot is {age_seconds:.0f}s old, generating new one...")
                    snapshot = self.signal_service.get_current_signal_snapshot(user_id)
            
            return snapshot
        except Exception as e:
            print(f"Warning: Could not fetch signal snapshot: {e}")
            return None
    
    def _build_coach_input(
        self,
        user_id: str,
        scheduled_tasks: list[ScheduledTask],
        signal_snapshot: Optional[SignalSnapshot],
        current_time: datetime,
        ignored_count: int,
        do_not_disturb: bool
    ) -> CoachInput:
        """
        Build the CoachInput from all available data.
        
        Args:
            user_id: The user's unique identifier
            scheduled_tasks: List of scheduled tasks
            signal_snapshot: Optional signal snapshot
            current_time: Current timestamp
            ignored_count: Number of ignored nudges
            do_not_disturb: DND flag
        
        Returns:
            A fully populated CoachInput
        """
        # Extract focus state from signals (or use defaults)
        if signal_snapshot is not None:
            focus_state = FocusState(
                state=signal_snapshot.focus_state,
                score=signal_snapshot.focus_score
            )
            # Extract fatigue data from signals
            fatigue_state = FatigueState(
                state=signal_snapshot.fatigue_state,
                score=signal_snapshot.fatigue_score
            )
        else:
            # Default to neutral state if no signals available
            focus_state = FocusState(state="Drifting", score=0.5)
            fatigue_state = FatigueState(state="Moderate", score=0.3)
        affective_state = "engaged"  # Mock value
        
        # Determine if user is late (simple check)
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
            signals=signal_snapshot  # Pass the full snapshot for future use
        )
    
    def _check_if_late(
        self, 
        scheduled_tasks: list[ScheduledTask], 
        current_time: datetime
    ) -> bool:
        """
        Check if the user is late for any scheduled task.
        
        Args:
            scheduled_tasks: List of scheduled tasks
            current_time: Current timestamp
        
        Returns:
            True if user is late for any task, False otherwise
        """
        for task in scheduled_tasks:
            if current_time > task.start_time:
                return True
        return False
