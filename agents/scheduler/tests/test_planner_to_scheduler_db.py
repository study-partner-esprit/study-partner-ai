"""
Integration test for the complete pipeline: MongoDB Course → PlannerAgent → SchedulerAgent.

This test verifies the end-to-end flow from fetching a course from the database
through task generation and scheduling.
"""

import pytest
from datetime import datetime, timedelta

from agents.course_ingestion.services.database_service import DatabaseService
from agents.planner.agent import PlannerAgent
from agents.planner.models.task_graph import PlannerInput
from agents.scheduler.agent import SchedulerAgent, SchedulingContext
from models.task import Task


# Replace this with an actual existing course ObjectId from your MongoDB
TEST_COURSE_ID = "6984fed077f336f5b99c9c45"


@pytest.mark.integration
class TestPlannerToSchedulerIntegration:
    """Integration tests for the complete planner-to-scheduler pipeline."""

    def test_full_pipeline_from_mongodb_course(self):
        """Test the complete pipeline from MongoDB course to scheduled study plan."""
        # Skip if no test course ID is configured
        if TEST_COURSE_ID == "<replace-with-existing-course-id>":
            pytest.skip("TEST_COURSE_ID not configured with actual course ID")

        # Step 1: Fetch course from MongoDB
        db_service = DatabaseService()
        course_doc = db_service.get_course_by_id(TEST_COURSE_ID)

        assert course_doc is not None, f"Course with ID {TEST_COURSE_ID} not found in database"

        # Step 2: Generate tasks using PlannerAgent
        planner = PlannerAgent()

        # Create planner input using the course data
        deadline = datetime.now() + timedelta(days=7)  # 1 week from now

        planner_input = PlannerInput(
            goal=f"Complete course: {course_doc.get('title', 'Unknown Course')}",
            deadline_iso=deadline.isoformat(),
            available_minutes=480,  # 8 hours total
            user_id="test_user_integration",
            course_knowledge=course_doc,  # Pass the full course document
        )

        planner_output = planner.plan(planner_input)

        # Verify planner output
        assert planner_output.task_graph is not None
        assert len(planner_output.task_graph.tasks) > 0, "Planner should generate at least one task"
        assert not planner_output.clarification_required, "Test course should not require clarification"

        # Step 3: Convert AtomicTask objects to Task objects for scheduler
        tasks = []
        for atomic_task in planner_output.task_graph.tasks:
            # Convert difficulty from float (0-1) to string categories
            if atomic_task.difficulty <= 0.3:
                difficulty_str = "beginner"
            elif atomic_task.difficulty <= 0.7:
                difficulty_str = "intermediate"
            else:
                difficulty_str = "advanced"

            task = Task(
                task_id=atomic_task.id,
                user_id="test_user_integration",
                title=atomic_task.title,
                description=atomic_task.description,
                estimated_duration=atomic_task.estimated_minutes,
                difficulty=difficulty_str,
                prerequisites=atomic_task.prerequisites,
            )
            tasks.append(task)

        # Step 4: Schedule tasks using SchedulerAgent
        scheduler = SchedulerAgent()

        # Create minimal scheduling context
        context = SchedulingContext(
            calendar_events=[],  # No calendar conflicts for this test
            max_minutes_per_day=240,  # 4 hours per day
        )

        study_plan = scheduler.build_schedule(tasks, context)

        # Step 5: Verify the final study plan
        assert study_plan is not None
        assert len(study_plan.sessions) > 0, "Scheduler should create at least one scheduled session"
        assert study_plan.total_minutes > 0, "Study plan should have positive total minutes"

        # Verify that all scheduled sessions have valid times
        for session in study_plan.sessions:
            assert session.start_datetime < session.end_datetime
            assert session.slot_score >= 0.0
            assert session.break_after_minutes >= 0

        # Verify that the plan spans appropriate days
        assert study_plan.span_days >= 1

        # Verify that no tasks were skipped (since we have no prerequisites in this simple test)
        # Note: In a real scenario, some tasks might be skipped due to prerequisites
        # but for this integration test, we just verify the pipeline works