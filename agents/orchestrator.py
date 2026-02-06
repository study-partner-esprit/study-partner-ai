"""
Orchestrator that integrates course ingestion with study planning.

This module coordinates the flow between:
1. Course ingestion agent (processes PDFs into structured course JSON)
2. Planner agent (decomposes learning goals into atomic tasks using course knowledge)
"""

from agents.course_ingestion.agent import ingest_course
from agents.course_ingestion.services.database_service import DatabaseService
from agents.planner.agent import PlannerAgent
from agents.planner.models.task_graph import PlannerInput
from agents.scheduler.agent import SchedulerAgent, SchedulingContext
from models.task import Task
from datetime import datetime, timedelta
import uuid


def run_study_planner(
    pdf_paths: list[str], learning_goal: str, available_time: int
) -> dict:
    """
    Orchestrate the complete study planning pipeline.

    Args:
        pdf_paths: List of paths to PDF course materials
        learning_goal: The learning goal to decompose into tasks
        available_time: Available study time in minutes

    Returns:
        Dictionary containing the planner output with task graph
    """
    # Step 1: Ingest the course materials
    print("📚 Ingesting course materials...")
    course_title = "Course Materials"  # Simple title for ingested content
    try:
        course_id = ingest_course(course_title, pdf_paths)
        print(f"✅ Course ingested with ID: {course_id}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"PDF file not found: {e}")
    except Exception as e:
        raise ValueError(f"Failed to ingest course materials: {e}")

    # Step 2: Retrieve the normalized course JSON from MongoDB
    print("🔍 Retrieving course data from database...")
    db = DatabaseService()
    course_data = db.get_course_by_id(course_id)

    print("✅ Course data retrieved successfully")

    # Step 3: Prepare planner input with course knowledge
    print("🎯 Preparing study plan...")

    # Create a deadline 7 days from now (arbitrary but reasonable)
    deadline = datetime.now() + timedelta(days=7)

    planner_input = PlannerInput(
        goal=learning_goal,
        deadline_iso=deadline.isoformat(),
        available_minutes=available_time,
        user_id=str(uuid.uuid4()),  # Generate a unique user ID for this session
        course_knowledge=course_data,  # Pass the full course JSON as context
    )

    # Step 4: Run the planner agent
    print("🧠 Generating study plan...")
    planner_agent = PlannerAgent()
    planner_output = planner_agent.plan(planner_input)

    print("✅ Study plan generated successfully")

    # Step 5: Save the study plan to database
    print("💾 Saving study plan to database...")
    study_plan_data = planner_output.model_dump()
    study_plan_data["course_id"] = course_id  # Link to the course
    study_plan_data["created_at"] = datetime.now().isoformat()

    study_plan_id = db.save_study_plan(study_plan_data)
    print(f"✅ Study plan saved with ID: {study_plan_id}")

    # Step 6: Return the result with study plan ID
    result = planner_output.model_dump()
    result["study_plan_id"] = study_plan_id
    result["course_id"] = course_id
    return result

def run_full_study_workflow(
    pdf_paths: list[str], learning_goal: str, available_time: int
) -> dict:
    """
    Orchestrate the complete study workflow from PDF to scheduled tasks.

    Args:
        pdf_paths: List of paths to PDF course materials
        learning_goal: The learning goal to decompose into tasks
        available_time: Available study time in minutes

    Returns:
        Dictionary containing planner output, scheduler output, and metadata
    """
    # Step 1-4: Run the planner pipeline (reuse existing logic)
    planner_result = run_study_planner(pdf_paths, learning_goal, available_time)

    # Step 5: Extract tasks from planner output and convert to Task objects
    print("🔄 Converting planner tasks to scheduler format...")
    planner_output = planner_result
    task_graph = planner_output.get("task_graph", {})
    atomic_tasks = task_graph.get("tasks", [])

    tasks = []
    for atomic_task in atomic_tasks:
        # Convert difficulty from float (0-1) to string categories
        difficulty_float = atomic_task.get("difficulty", 0.5)
        if difficulty_float <= 0.3:
            difficulty_str = "beginner"
        elif difficulty_float <= 0.7:
            difficulty_str = "intermediate"
        else:
            difficulty_str = "advanced"

        task = Task(
            task_id=atomic_task["id"],
            user_id="full_workflow_user",  # Fixed user ID for this workflow
            title=atomic_task["title"],
            description=atomic_task["description"],
            estimated_duration=atomic_task["estimated_minutes"],
            difficulty=difficulty_str,
            prerequisites=atomic_task.get("prerequisites", []),
        )
        tasks.append(task)

    print(f"✅ Converted {len(tasks)} tasks for scheduling")

    # Step 6: Run the scheduler
    print("📅 Scheduling study sessions...")
    scheduler = SchedulerAgent()

    # Create scheduling context with reasonable defaults
    context = SchedulingContext(
        calendar_events=[],  # No calendar conflicts for this demo
        max_minutes_per_day=240,  # 4 hours per day
    )

    study_plan = scheduler.build_schedule(tasks, context)

    print("✅ Study sessions scheduled successfully")
    print(f"📊 Total scheduled time: {study_plan.total_minutes} minutes")
    print(f"📅 Plan spans {study_plan.span_days} days")
    print(f"🎯 Number of scheduled sessions: {len(study_plan.sessions)}")

    # Step 7: Save the scheduled sessions to the task_scheduling collection
    print("💾 Saving scheduled sessions to database...")
    db = DatabaseService()
    scheduler_data = study_plan.model_dump()
    scheduler_data["course_id"] = planner_result.get("course_id")
    scheduling_id = db.save_scheduled_sessions(planner_result.get("study_plan_id"), scheduler_data)
    print(f"✅ Scheduled sessions saved with ID: {scheduling_id}")

    # Step 8: Return comprehensive result
    result = {
        "planner_output": planner_output,
        "scheduler_output": study_plan.model_dump(),
        "metadata": {
            "course_id": planner_result.get("course_id"),
            "study_plan_id": planner_result.get("study_plan_id"),
            "scheduling_id": scheduling_id,
            "total_tasks": len(tasks),
            "scheduled_sessions": len(study_plan.sessions),
            "total_scheduled_minutes": study_plan.total_minutes,
            "plan_span_days": study_plan.span_days,
            "fallback_used": study_plan.fallback_used,
            "skipped_tasks": study_plan.skipped_tasks,
        }
    }

    return result

    # Step 8: Return comprehensive result
    result = {
        "planner_output": planner_output,
        "scheduler_output": study_plan.model_dump(),
        "metadata": {
            "course_id": planner_result.get("course_id"),
            "study_plan_id": planner_result.get("study_plan_id"),
            "total_tasks": len(tasks),
            "scheduled_sessions": len(study_plan.sessions),
            "total_scheduled_minutes": study_plan.total_minutes,
            "plan_span_days": study_plan.span_days,
            "fallback_used": study_plan.fallback_used,
            "skipped_tasks": study_plan.skipped_tasks,
        }
    }

    return result

# Convenience function for testing
def run_study_planner_with_course_id(
    course_id: str, learning_goal: str, available_time: int
) -> dict:
    """
    Alternative entry point that uses an existing course ID instead of ingesting PDFs.

    Args:
        course_id: Existing course ID from MongoDB
        learning_goal: The learning goal to decompose into tasks
        available_time: Available study time in minutes

    Returns:
        Dictionary containing the planner output with task graph
    """
    # Step 1: Retrieve the normalized course JSON from MongoDB
    print("🔍 Retrieving course data from database...")
    db = DatabaseService()
    course_data = db.get_course(course_id)

    if not course_data:
        raise ValueError(f"Could not retrieve course data for ID: {course_id}")

    print("✅ Course data retrieved successfully")

    # Step 2: Prepare planner input with course knowledge
    print("🎯 Preparing study plan...")

    # Create a deadline 7 days from now
    deadline = datetime.now() + timedelta(days=7)

    planner_input = PlannerInput(
        goal=learning_goal,
        deadline_iso=deadline.isoformat(),
        available_minutes=available_time,
        user_id=str(uuid.uuid4()),
        course_knowledge=course_data,
    )

    # Step 3: Run the planner agent
    print("🧠 Generating study plan...")
    planner_agent = PlannerAgent()
    planner_output = planner_agent.plan(planner_input)

    print("✅ Study plan generated successfully")

    # Step 4: Save the study plan to database
    print("💾 Saving study plan to database...")
    study_plan_data = planner_output.model_dump()
    study_plan_data["course_id"] = course_id  # Link to the course
    study_plan_data["created_at"] = datetime.now().isoformat()

    db = DatabaseService()
    study_plan_id = db.save_study_plan(study_plan_data)
    print(f"✅ Study plan saved with ID: {study_plan_id}")

    # Step 5: Return the result with study plan ID
    result = planner_output.model_dump()
    result["study_plan_id"] = study_plan_id
    result["course_id"] = course_id
    return result
