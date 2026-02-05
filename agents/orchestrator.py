"""
Orchestrator that integrates course ingestion with study planning.

This module coordinates the flow between:
1. Course ingestion agent (processes PDFs into structured course JSON)
2. Planner agent (decomposes learning goals into atomic tasks using course knowledge)
"""

from agents.course_ingestion.agent import ingest_course
from agents.course_ingestion.services.database_service import DatabaseService
from agents.planner.agent import PlannerAgent
from agents.planner.models.task_graph import PlannerInput, PlannerOutput
from datetime import datetime, timedelta
import uuid


def run_study_planner(
    pdf_paths: list[str],
    learning_goal: str,
    available_time: int
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
    course_data = db.get_course(course_id)

    if not course_data:
        raise ValueError(f"Could not retrieve course data for ID: {course_id}")

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
        course_knowledge=course_data  # Pass the full course JSON as context
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


# Convenience function for testing
def run_study_planner_with_course_id(
    course_id: str,
    learning_goal: str,
    available_time: int
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
        course_knowledge=course_data
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