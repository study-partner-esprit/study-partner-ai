"""Schedule orchestration endpoints."""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from deps import get_schedule_orchestrator
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class SchedulerRequest(BaseModel):
    user_id: str
    tasks: List[dict]  # List of task objects from Node.js
    calendar_events: Optional[List[dict]] = []
    max_minutes_per_day: int = 240
    allow_late_night: bool = False


class RescheduleRequest(BaseModel):
    user_id: str
    reason: str = "manual"


class SchedulerOptimizeRequest(BaseModel):
    user_id: str
    reason: str = "manual_optimize"


class SchedulerApplyCoachActionRequest(BaseModel):
    user_id: str
    coach_action: Dict[str, Any]


def mix_tasks_intelligently(tasks: List) -> List:
    """
    Mix tasks from different subjects/plans to avoid scheduling all tasks
    from the same subject consecutively.

    Strategy:
    1. Group tasks by tags/subject (first tag indicates subject/course)
    2. Interleave tasks from different groups (round-robin)
    3. Maintain tasks without tags at the end

    Args:
        tasks: List of SchedulerTask objects

    Returns:
        Reordered list with mixed tasks
    """
    # Group tasks by their primary tag (subject/course)
    groups = defaultdict(list)
    no_tag_tasks = []

    for task in tasks:
        if task.tags and len(task.tags) > 0:
            # Use first tag as grouping key (usually subject/course)
            primary_tag = task.tags[0]
            groups[primary_tag].append(task)
        else:
            no_tag_tasks.append(task)

    # If there's only one group or no groups, return original order
    if len(groups) <= 1:
        return tasks

    # Mix tasks using round-robin from each group
    mixed_tasks = []
    group_lists = list(groups.values())
    max_length = max(len(group) for group in group_lists)

    for i in range(max_length):
        for group in group_lists:
            if i < len(group):
                mixed_tasks.append(group[i])

    # Add tasks without tags at the end
    mixed_tasks.extend(no_tag_tasks)

    logger.info(
        f"Task mixing: {len(groups)} groups found, mixed {len(mixed_tasks)} tasks"
    )
    for tag, group in groups.items():
        logger.info(f"  Group '{tag}': {len(group)} tasks")

    return mixed_tasks


@router.post("/api/ai/scheduler/schedule")
async def schedule_tasks(request: SchedulerRequest):
    """
    Schedule existing tasks using the AI scheduler agent.

    Args:
        request: SchedulerRequest with tasks and scheduling constraints

    Returns:
        Schedule with sessions, total time, and metadata
    """
    try:
        from agents.scheduler.agent import SchedulerAgent, SchedulingContext
        from models.task import Task as SchedulerTask

        # Convert Node.js tasks to scheduler Task objects
        scheduler_tasks = []
        for task in request.tasks:
            # Map priority to difficulty (0-1 scale)
            priority = task.get("priority", "medium")
            if priority == "low":
                difficulty = 0.3
            elif priority == "high":
                difficulty = 0.8
            else:
                difficulty = 0.5

            scheduler_task = SchedulerTask(
                task_id=str(task.get("_id", task.get("id", ""))),
                user_id=request.user_id,
                title=task.get("title", "Untitled Task"),
                description=task.get("description", ""),
                priority=priority,
                difficulty=str(difficulty),
                estimated_duration=task.get("estimatedTime", 30),
                status=task.get("status", "todo"),
                tags=task.get("tags", []),
                prerequisites=task.get("prerequisites", []),
            )
            scheduler_tasks.append(scheduler_task)

        # Mix tasks from different subjects/courses intelligently
        mixed_tasks = mix_tasks_intelligently(scheduler_tasks)

        # Create  scheduling context
        context = SchedulingContext(
            calendar_events=request.calendar_events,
            max_minutes_per_day=request.max_minutes_per_day,
            allow_late_night=request.allow_late_night,
        )

        # Build schedule with mixed tasks
        scheduler = SchedulerAgent()
        study_plan = scheduler.build_schedule(
            tasks=mixed_tasks,
            context=context,
        )

        # Convert sessions to frontend format
        sessions = []
        for session in study_plan.sessions:
            sessions.append(
                {
                    "taskId": session.task_id,
                    "title": next(
                        (
                            t.title
                            for t in scheduler_tasks
                            if t.task_id == session.task_id
                        ),
                        "Unknown",
                    ),
                    "startTime": session.start_datetime.isoformat(),
                    "endTime": session.end_datetime.isoformat(),
                    "estimatedMinutes": int(
                        (session.end_datetime - session.start_datetime).total_seconds()
                        / 60
                    ),
                    "slotScore": session.slot_score,
                }
            )

        return {
            "success": True,
            "schedule": {
                "sessions": sessions,
                "totalMinutes": study_plan.total_minutes,
                "spanDays": study_plan.span_days,
                "skippedTasks": study_plan.skipped_tasks,
                "fallbackUsed": study_plan.fallback_used,
            },
        }

    except Exception as e:
        logger.error(f"Task scheduling failed: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to schedule tasks: {str(e)}"
        )


@router.post("/api/ai/scheduler/reschedule")
async def reschedule(request: RescheduleRequest):
    """
    Trigger a re-schedule for a user (e.g. after a long break or plan change).
    Returns updated schedule.
    """
    try:
        from agents.course_ingestion.services.database_service import DatabaseService

        db_svc = DatabaseService()
        study_plan = db_svc.get_latest_study_plan(request.user_id)
        if study_plan is None:
            raise HTTPException(status_code=404, detail="No study plan found")
        return {
            "status": "ok",
            "message": "Reschedule queued",
            "user_id": request.user_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/ai/scheduler/apply-coach-action")
async def apply_coach_action_to_schedule(request: SchedulerApplyCoachActionRequest):
    """Apply explicit coach action schedule changes for a user."""
    try:
        from agents.coach.models.schemas import CoachAction

        coach_action = CoachAction(**request.coach_action)
        result = get_schedule_orchestrator().process_coach_action(
            coach_action=coach_action,
            user_id=request.user_id,
            current_time=datetime.now(),
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/ai/scheduler/status/{user_id}")
async def scheduler_status(user_id: str):
    """Get current schedule status for a user."""
    try:
        status = get_schedule_orchestrator().get_schedule_status(user_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/ai/scheduler/optimize")
async def optimize_schedule(request: SchedulerOptimizeRequest):
    """Run schedule optimization for a user and persist snapshot."""
    try:
        result = get_schedule_orchestrator().optimize_schedule(
            request.user_id, reason=request.reason
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/ai/vector/rebuild/{course_id}")
async def rebuild_vector_index(course_id: str):
    """Attempt to load/rebuild a course vector index from persisted stores."""
    try:
        from services.vector_store.adapter import get_vector_store

        store = get_vector_store()
        loaded = store.load_course(course_id)
        if not loaded:
            raise HTTPException(status_code=404, detail="Course index not found")

        return {
            "status": "ok",
            "course_id": course_id,
            "loaded": True,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/ai/vector/status/{course_id}")
async def vector_index_status(course_id: str):
    """Get in-memory status of a course vector index."""
    try:
        from services.vector_store.adapter import get_vector_store

        store = get_vector_store()
        loaded = store.load_course(course_id)
        return {
            "status": "ok",
            "course_id": course_id,
            "loaded": loaded,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
