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
from agents.scheduler.services.schedule_updater import ScheduleUpdater
from agents.coach.agent import run_coach
from agents.coach.models.schemas import CoachInput, FocusState
from agents.evaluator.src.evaluator.evaluator_agent import EvaluatorAgent
from models.task import Task
from datetime import datetime, timedelta
import uuid


def run_study_planner(
    pdf_paths: list[str], learning_goal: str = None, available_time: int = None,
    user_id: str = None,
) -> dict:
    """
    Orchestrate the complete study planning pipeline.

    Args:
        pdf_paths: List of paths to PDF course materials
        learning_goal: Optional learning goal (if not provided, will be derived from course)
        available_time: Available study time in minutes (default: 480 = 8 hours)
        user_id: Authenticated user ID (required for history/pacing persistence)

    Returns:
        Dictionary containing the planner output with task graph
    """
    if user_id is None:
        raise ValueError("user_id is required – pass the authenticated user's ID")
    if available_time is None:
        available_time = 480  # Default 8 hours
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
        goal=learning_goal,  # Can be None - planner will derive from course
        deadline_iso=deadline.isoformat(),
        available_minutes=available_time,
        user_id=user_id,
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
    study_plan_data["user_id"] = user_id
    study_plan_data["created_at"] = datetime.now().isoformat()

    study_plan_id = db.save_study_plan(study_plan_data)
    print(f"✅ Study plan saved with ID: {study_plan_id}")

    # Step 6: Return the result with study plan ID
    result = planner_output.model_dump()
    result["study_plan_id"] = study_plan_id
    result["course_id"] = course_id
    return result

def run_full_study_workflow(
    pdf_paths: list[str], learning_goal: str = None, available_time: int = None,
    user_id: str = None,
) -> dict:
    """
    Orchestrate the complete study workflow from PDF to scheduled tasks.

    Args:
        pdf_paths: List of paths to PDF course materials
        learning_goal: Optional learning goal (if not provided, will be derived from course)
        available_time: Available study time in minutes (default: 480 = 8 hours)
        user_id: Authenticated user ID (required for history/pacing persistence)

    Returns:
        Dictionary containing planner output, scheduler output, and metadata
    """
    if user_id is None:
        raise ValueError("user_id is required – pass the authenticated user's ID")
    if available_time is None:
        available_time = 480  # Default 8 hours

    # Step 1-4: Run the planner pipeline (reuse existing logic)
    planner_result = run_study_planner(pdf_paths, learning_goal, available_time, user_id=user_id)

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
            user_id=user_id,
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
    db = DatabaseService()
    
    # Generate calendar events from user's availability slots for the next 30 days
    calendar_events = []
    if user_id:
        availability_slots = db.get_availability_slots(user_id)
        current_date = datetime.now()
        
        # Map day names to weekday integers (0 = Monday, 6 = Sunday)
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        
        for i in range(30):
            check_date = current_date + timedelta(days=i)
            check_weekday = check_date.weekday()
            
            for slot in availability_slots:
                slot_weekday = day_map.get(slot.get("dayOfWeek"))
                if slot_weekday == check_weekday:
                    # Found a blocked slot for this day
                    # Parse start and end times
                    start_str = slot.get("startTime", "00:00")
                    end_str = slot.get("endTime", "23:59")
                    
                    try:
                        start_h, start_m = map(int, start_str.split(":"))
                        end_h, end_m = map(int, end_str.split(":"))
                        
                        start_dt = check_date.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
                        end_dt = check_date.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
                        
                        calendar_events.append({
                            "start": start_dt.isoformat(),
                            "end": end_dt.isoformat(),
                            "title": slot.get("label", "Blocked")
                        })
                    except Exception as e:
                        print(f"Warning: Could not parse time for slot {slot}: {e}")

    # Create scheduling context
    context = SchedulingContext(
        calendar_events=calendar_events,
        max_minutes_per_day=240,  # 4 hours per day
        user_id=user_id
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

# Convenience function for testing
def run_study_planner_with_course_id(
    course_id: str, learning_goal: str, available_time: int,
    user_id: str = None,
) -> dict:
    """
    Alternative entry point that uses an existing course ID instead of ingesting PDFs.

    Args:
        course_id: Existing course ID from MongoDB
        learning_goal: The learning goal to decompose into tasks
        available_time: Available study time in minutes
        user_id: Authenticated user ID (required for history/pacing persistence)

    Returns:
        Dictionary containing the planner output with task graph
    """
    if user_id is None:
        raise ValueError("user_id is required – pass the authenticated user's ID")
    # Step 1: Retrieve the normalized course JSON from MongoDB
    print("🔍 Retrieving course data from database...")
    db = DatabaseService()
    course_data = db.get_course_by_id(course_id)

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
        user_id=user_id,
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
    study_plan_data["user_id"] = user_id
    study_plan_data["created_at"] = datetime.now().isoformat()

    db = DatabaseService()
    study_plan_id = db.save_study_plan(study_plan_data)
    print(f"✅ Study plan saved with ID: {study_plan_id}")

    # Step 5: Return the result with study plan ID
    result = planner_output.model_dump()
    result["study_plan_id"] = study_plan_id
    result["course_id"] = course_id
    return result


def run_coaching_session(
    focus_state: str,
    focus_score: float,
    fatigue_probability: float,
    affective_state: str,
    ignored_count: int = 0,
    do_not_disturb: bool = False,
    is_late: bool = False
) -> dict:
    """
    Run a coaching session with real-time student monitoring.

    This function simulates a coaching session where the Coach agent:
    1. Monitors student's current state (focus, fatigue, affect)
    2. Provides appropriate coaching interventions
    3. Can autonomously modify the schedule (add breaks, reschedule tasks)

    Args:
        focus_state: Current focus state ("Focused", "Drifting", "Lost")
        focus_score: Focus score (0.0 to 1.0)
        fatigue_probability: Probability of fatigue (0.0 to 1.0)
        affective_state: Current emotional state
        ignored_count: How many times coaching was ignored
        do_not_disturb: Whether DND mode is active

    Returns:
        Dictionary with coaching action and any schedule changes applied
    """
    print("🤖 Starting coaching session...")

    # Create Coach input with current student state
    coach_input = CoachInput(
        scheduled_tasks=[],  # Will be populated by Coach agent from DB
        current_time=datetime.now(),
        focus_state=FocusState(state=focus_state, score=focus_score),
        fatigue_probability=fatigue_probability,
        affective_state=affective_state,
        ignored_count=ignored_count,
        do_not_disturb=do_not_disturb,
        is_late=is_late
    )

    # Run Coach agent
    print("🧠 Coach analyzing student state...")
    coach_action = run_coach(coach_input)

    result = {
        "coach_action": coach_action.model_dump(),
        "schedule_modified": False,
        "modifications": []
    }

    # Check if Coach requested schedule changes
    if coach_action.schedule_changes:
        print("📅 Coach requested schedule modification...")

        # Apply schedule changes
        schedule_updater = ScheduleUpdater()
        success = schedule_updater.apply_schedule_change(coach_action.schedule_changes)

        if success:
            result["schedule_modified"] = True
            result["modifications"].append({
                "action": coach_action.schedule_changes.action,
                "details": coach_action.schedule_changes.model_dump()
            })
            print(f"✅ Schedule updated: {coach_action.schedule_changes.action}")
        else:
            print("❌ Failed to apply schedule changes")

    # Log the coaching intervention
    print(f"💬 Coach Action: {coach_action.action_type}")
    if coach_action.message:
        print(f"   Message: {coach_action.message}")
    print(f"   Reasoning: {coach_action.reasoning}")

    return result


def run_evaluation_step(
    session_id: str = None,
    task_id: str = None,
    task_title: str = "",
    task_description: str = "",
    task_details: str = "",
    student_answer: str = None,
) -> dict:
    """
    Run an evaluation step to assess student mastery of a task.
    This links the Evaluator Agent into the learning loop.

    If session_id is None, it starts a new evaluation session for the task.
    If student_answer is provided, it processes the answer.
    """
    evaluator = EvaluatorAgent()
    
    # We maintain sessions in the evaluator instance locally, but for a true 
    # stateless orchestrator we would hydrate the session from the DB.
    # For now, we assume the EvaluatorAgent manages its own session state in memory.
    
    if not session_id:
        print(f"🎓 Starting new evaluation session for task: {task_title}")
        result = evaluator.start_session(
            task_title=task_title,
            task_description=task_description,
            task_details=task_details
        )
        return {
            "action": "STARTED",
            "session_id": result["session_id"],
            "question": result["question"],
        }
    
    if student_answer:
        print(f"🎓 Evaluating student answer for session: {session_id}")
        result = evaluator.handle_user_answer(session_id, student_answer)
        
        # Here we integrate with the scheduling loop based on the result
        updater = ScheduleUpdater()
        
        if result.get("state") == "MASTERY_CONFIRMED":
            print(f"✅ Student mastered the task! Moving to next task.")
            # Spaced repetition & Adaptive Time Estimation
            updater.handle_evaluation_result(task_id, task_title, result)
            
        elif result.get("state") == "FAILED":
            print(f"❌ Student failed the task. Triggering reschedule...")
            # Reschedule a review session
            updater.handle_evaluation_result(task_id, task_title, result)
            
        elif result.get("state") == "CONTINUE":
            print(f"🔄 Continuing evaluation...")
            
        return result
    
    return {"error": "Invalid parameters"}