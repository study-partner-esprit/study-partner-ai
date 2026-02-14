from datetime import datetime, timedelta
from typing import List, Set
import logging

from agents.scheduler.models.schedule_schema import (
    StudyPlan,
    ScheduledSession,
)
from agents.scheduler.services.calendar_normalizer import normalize_busy_slots
from agents.scheduler.services.slot_generator import generate_free_slots
from agents.scheduler.services.scheduling_heuristics import score_slot
from models.task import Task

logger = logging.getLogger(__name__)


class SchedulingContext:

    def __init__(
        self,
        calendar_events: list,
        historical_productivity: list | None = None,
        fatigue_predictions: list | None = None,
        max_minutes_per_day: int = 240,
        allow_late_night: bool = False,
    ):
        self.calendar_events = calendar_events
        self.historical_productivity = historical_productivity or []
        self.fatigue_predictions = fatigue_predictions or []
        self.max_minutes_per_day = max_minutes_per_day
        self.allow_late_night = allow_late_night


class SchedulerAgent:
    """
    Scheduler agent with support for:
    - Prerequisite-aware scheduling
    - Multi-day scheduling
    - Fallback Pomodoro-style scheduling
    """
    
    WORK_START_HOUR = 8
    WORK_END_HOUR = 22
    FALLBACK_STUDY_MINUTES = 25
    FALLBACK_BREAK_MINUTES = 5

    def build_schedule(
        self,
        tasks: List[Task],
        context: SchedulingContext,
    ) -> StudyPlan:
        """
        Build a multi-day study schedule with prerequisite awareness.
        
        Args:
            tasks: List of tasks to schedule
            context: Scheduling context with calendar and constraints
            
        Returns:
            StudyPlan with scheduled sessions, skipped tasks, and metadata
        """
        sessions: List[ScheduledSession] = []
        skipped_tasks: Set[str] = set()
        scheduled_task_ids: Set[str] = set()
        total_minutes = 0
        fallback_used = False
        
        # Build title→id mapping for prerequisite resolution
        title_to_id = {task.title: task.id for task in tasks}
        
        # Start from tomorrow 8:00 AM
        current_date = datetime.now().replace(
            hour=self.WORK_START_HOUR,
            minute=0,
            second=0,
            microsecond=0,
        )
        
        # If it's too late today, start tomorrow
        if datetime.now().hour >= self.WORK_END_HOUR:
            current_date += timedelta(days=1)
        
        max_days = 30  # Prevent infinite loops
        day_count = 0
        task_index = 0
        
        while task_index < len(tasks) and day_count < max_days:
            day_sessions, day_total, day_fallback = self._schedule_day(
                tasks=tasks,
                task_index=task_index,
                current_date=current_date,
                context=context,
                scheduled_task_ids=scheduled_task_ids,
                skipped_tasks=skipped_tasks,
                title_to_id=title_to_id,
            )
            
            sessions.extend(day_sessions)
            total_minutes += day_total
            if day_fallback:
                fallback_used = True
            
            # Count how many tasks were actually scheduled this day
            tasks_scheduled_today = len(day_sessions)
            if tasks_scheduled_today == 0:
                # No progress made, skip to next day
                day_count += 1
                current_date += timedelta(days=1)
                continue
            
            # Move to next day
            current_date += timedelta(days=1)
            day_count += 1
            
            # Move task index forward only if we've scheduled something
            while (
                task_index < len(tasks)
                and tasks[task_index].id in scheduled_task_ids
            ):
                task_index += 1
        
        # Any remaining tasks couldn't be scheduled
        for task in tasks:
            if task.id not in scheduled_task_ids:
                skipped_tasks.add(task.id)
        
        span_days = min(day_count, len(set(s.start_datetime.date() for s in sessions)))
        span_days = max(1, span_days)
        
        return StudyPlan(
            sessions=sessions,
            total_minutes=total_minutes,
            fallback_used=fallback_used,
            skipped_tasks=list(skipped_tasks),
            span_days=span_days,
        )
    
    def _schedule_day(
        self,
        tasks: List[Task],
        task_index: int,
        current_date: datetime,
        context: SchedulingContext,
        scheduled_task_ids: Set[str],
        skipped_tasks: Set[str],
        title_to_id: dict,
    ) -> tuple[List[ScheduledSession], int, bool]:
        """
        Schedule tasks for a single day.
        
        Returns:
            (sessions, total_minutes, fallback_used)
        """
        day_start = current_date.replace(
            hour=self.WORK_START_HOUR,
            minute=0,
            second=0,
            microsecond=0,
        )
        day_end = day_start.replace(hour=self.WORK_END_HOUR)
        
        # Normalize busy slots for this day
        day_events = [
            ev for ev in context.calendar_events
            if datetime.fromisoformat(ev["start"]).date() == day_start.date()
        ]
        busy_slots = normalize_busy_slots(day_events)
        
        # Generate free slots for this day
        free_slots = generate_free_slots(
            day_start=day_start,
            day_end=day_end,
            busy_slots=busy_slots,
        )
        
        # Score slots
        for slot in free_slots:
            slot.score = score_slot(
                slot,
                context.historical_productivity,
                context.allow_late_night,
            )
        
        # Sort by score descending
        free_slots.sort(key=lambda s: s.score, reverse=True)
        
        sessions: List[ScheduledSession] = []
        daily_minutes = 0
        slot_index = 0
        fallback_used = False
        
        # Try to schedule tasks in order
        working_task_index = task_index
        
        while (
            working_task_index < len(tasks)
            and daily_minutes < context.max_minutes_per_day
            and slot_index < len(free_slots)
        ):
            task = tasks[working_task_index]
            
            # Check if task has unmet prerequisites
            if not self._prerequisites_met(task, scheduled_task_ids, title_to_id):
                logger.warning(
                    f"Skipping task {task.id}: unmet prerequisites {task.prerequisites}"
                )
                skipped_tasks.add(task.id)
                working_task_index += 1
                continue
            
            # Skip if already scheduled
            if task.id in scheduled_task_ids:
                working_task_index += 1
                continue
            
            # Try to fit task in available slots
            slot = free_slots[slot_index]
            duration = task.estimated_minutes
            
            start = slot.start
            end = start + timedelta(minutes=duration)
            
            # Check if task fits in current slot
            if end > slot.end:
                # Try next slot
                slot_index += 1
                continue
            
            # Task fits! Schedule it
            session = ScheduledSession(
                task_id=task.id,
                start_datetime=start,
                end_datetime=end,
                break_after_minutes=5,
                slot_score=slot.score,
                scheduled=True,
            )
            sessions.append(session)
            scheduled_task_ids.add(task.id)
            daily_minutes += duration
            
            # Move slot cursor past break time
            slot.start = end + timedelta(minutes=5)
            working_task_index += 1
        
        # If no progress and we have tasks, use fallback Pomodoro schedule
        if len(sessions) == 0 and working_task_index < len(tasks):
            sessions, daily_minutes, fallback_used = self._apply_fallback_schedule(
                tasks=tasks,
                task_index=working_task_index,
                day_start=day_start,
                day_end=day_end,
                context=context,
                scheduled_task_ids=scheduled_task_ids,
                skipped_tasks=skipped_tasks,
                title_to_id=title_to_id,
            )
        
        return sessions, daily_minutes, fallback_used
    
    def _prerequisites_met(
        self,
        task: Task,
        scheduled_task_ids: Set[str],
        title_to_id: dict,
    ) -> bool:
        """Check if all prerequisites for a task have been scheduled."""
        if not task.prerequisites:
            return True
        
        for prereq in task.prerequisites:
            # Prerequisite can be either a title (string) or ID (UUID)
            # Try to resolve title to ID if it's not already an ID
            prereq_id = title_to_id.get(prereq, prereq)
            
            if prereq_id not in scheduled_task_ids:
                return False
        
        return True
    
    def _apply_fallback_schedule(
        self,
        tasks: List[Task],
        task_index: int,
        day_start: datetime,
        day_end: datetime,
        context: SchedulingContext,
        scheduled_task_ids: Set[str],
        skipped_tasks: Set[str],
        title_to_id: dict,
    ) -> tuple[List[ScheduledSession], int, bool]:
        """
        Apply fallback Pomodoro-style schedule when no free slots exist.
        
        Returns:
            (sessions, total_minutes, fallback_used=True)
        """
        sessions: List[ScheduledSession] = []
        daily_minutes = 0
        current_time = day_start
        
        working_task_index = task_index
        
        while (
            working_task_index < len(tasks)
            and daily_minutes < context.max_minutes_per_day
            and current_time + timedelta(minutes=self.FALLBACK_STUDY_MINUTES) <= day_end
        ):
            task = tasks[working_task_index]
            
            # Check prerequisites
            if not self._prerequisites_met(task, scheduled_task_ids, title_to_id):
                logger.warning(
                    f"Skipping task {task.id} in fallback: unmet prerequisites"
                )
                skipped_tasks.add(task.id)
                working_task_index += 1
                continue
            
            if task.id in scheduled_task_ids:
                working_task_index += 1
                continue
            
            # Schedule in Pomodoro chunks
            session_duration = min(
                task.estimated_minutes,
                self.FALLBACK_STUDY_MINUTES,
            )
            
            end_time = current_time + timedelta(minutes=session_duration)
            
            session = ScheduledSession(
                task_id=task.id,
                start_datetime=current_time,
                end_datetime=end_time,
                break_after_minutes=self.FALLBACK_BREAK_MINUTES,
                slot_score=0.5,  # Lower score for fallback
                scheduled=True,
            )
            sessions.append(session)
            scheduled_task_ids.add(task.id)
            daily_minutes += session_duration
            
            # Move to next slot (study + break)
            current_time = end_time + timedelta(
                minutes=self.FALLBACK_BREAK_MINUTES
            )
            
            working_task_index += 1
        
        return sessions, daily_minutes, len(sessions) > 0

