"""Study plan generation endpoints."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from deps import convert_objectid_to_str, get_planner_agent
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class PlannerRequest(BaseModel):
    user_id: str
    goal: str
    available_time_minutes: int
    start_date: Optional[datetime] = None
    course_id: Optional[str] = None
    calendar_events: Optional[list] = (
        []
    )  # User's blocked time slots from Node.js backend


class RecordCompletionRequest(BaseModel):
    user_id: str
    task_id: str
    estimated_minutes: int
    actual_minutes: int
    subject_tag: str = ""


@router.post("/api/ai/planner/create-plan")
async def create_study_plan(request: PlannerRequest):
    """
    Create a personalized study plan by generating tasks.

    This endpoint creates tasks directly instead of a separate study plan concept.
    Tasks are saved to the database and can be accessed via /api/v1/study/tasks

    Args:
        request: PlannerRequest with goal, time, and optional course

    Returns:
        List of generated tasks with metadata
    """
    try:
        from agents.planner.models.task_graph import PlannerInput

        # Fetch course documents if course_id provided
        course_knowledge = None
        if request.course_id:
            from agents.course_ingestion.services.database_service import (
                DatabaseService,
            )

            db_service = DatabaseService()
            course = db_service.get_course_by_id(request.course_id)
            if course:
                # Convert all ObjectIds to strings for JSON serialization
                course_knowledge = convert_objectid_to_str(course)
                print(
                    f"DEBUG: Course knowledge loaded for course_id {request.course_id}"
                )
                print(
                    f"DEBUG: Course has {len(course_knowledge.get('topics', []))} topics"
                )
                if (
                    course_knowledge.get("topics")
                    and len(course_knowledge["topics"]) > 0
                ):
                    first_topic = course_knowledge["topics"][0]
                    if "subtopics" in first_topic and len(first_topic["subtopics"]) > 0:
                        first_subtopic = first_topic["subtopics"][0]
                        has_tokenized = "tokenized_chunks" in first_subtopic
                        has_summary = "summary" in first_subtopic
                        print(
                            f"DEBUG: First subtopic has tokenized_chunks: {has_tokenized}, has summary: {has_summary}"
                        )
                        if has_tokenized:
                            print(
                                f"DEBUG: tokenized_chunks length: {len(first_subtopic['tokenized_chunks'])}"
                            )
                            print(
                                f"DEBUG: First chunk preview: {first_subtopic['tokenized_chunks'][0][:100]}..."
                            )
                        if has_summary:
                            print(
                                f"DEBUG: summary preview: {first_subtopic['summary'][:100]}..."
                            )

        # Create planner input
        if request.start_date:
            deadline_iso = request.start_date.isoformat()
        else:
            deadline_iso = (datetime.now() + timedelta(days=7)).isoformat()

        planner_input = PlannerInput(
            goal=request.goal,
            deadline_iso=deadline_iso,
            available_minutes=request.available_time_minutes,
            user_id=request.user_id,
            course_knowledge=course_knowledge,
        )

        print(
            f"DEBUG: Created planner input with goal: '{request.goal}', course_knowledge: {course_knowledge is not None}"
        )
        if course_knowledge:
            print(f"DEBUG: Course title: {course_knowledge.get('course_title', 'N/A')}")

        # Generate plan using planner agent
        planner_agent = get_planner_agent()
        plan_output = planner_agent.plan(planner_input)

        # Convert AtomicTasks to Task format for database
        tasks = []
        for atomic_task in plan_output.task_graph.tasks:
            # Map difficulty to priority
            if atomic_task.difficulty < 0.4:
                priority = "low"
            elif atomic_task.difficulty < 0.7:
                priority = "medium"
            else:
                priority = "high"

            task = {
                "title": atomic_task.title,
                "description": atomic_task.description,
                "priority": priority,
                "estimatedTime": atomic_task.estimated_minutes,
                "tags": (
                    [request.goal[:50]] if request.goal else []
                ),  # Use goal as a tag
            }
            tasks.append(task)

        return {
            "success": True,
            "tasks": tasks,
            "count": len(tasks),
            "total_time": sum(t["estimatedTime"] for t in tasks),
            "warning": plan_output.warning if hasattr(plan_output, "warning") else None,
        }

    except Exception as e:
        logger.error(f"Study plan creation failed: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to create study plan: {str(e)}"
        )


@router.get("/api/ai/planner/plans/{user_id}")
async def get_user_plans(user_id: str):
    """Get all study plans for a user."""
    try:
        from agents.planner.rag.prompt_builder import SchedulingService

        scheduling_service = SchedulingService()
        plans = scheduling_service.get_user_plans(user_id)
        return {"plans": plans}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch plans: {str(e)}")


@router.post("/api/ai/planner/record-completion")
async def record_task_completion(request: RecordCompletionRequest):
    """
    Record actual vs estimated task completion time to improve future pacing.
    """
    try:
        from agents.planner.memory.pacing_store import PacingStore

        store = PacingStore()
        store.record_task_completion(
            user_id=request.user_id,
            task_id=request.task_id,
            estimated_minutes=request.estimated_minutes,
            actual_minutes=request.actual_minutes,
            subject_tag=request.subject_tag,
        )
        factor = store.get_user_pacing_factor(request.user_id, request.subject_tag)
        return {"status": "ok", "new_pacing_factor": factor}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
