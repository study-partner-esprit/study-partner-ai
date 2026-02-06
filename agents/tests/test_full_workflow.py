"""
End-to-end integration test for the complete study workflow.

Tests the full pipeline from PDF ingestion → course processing → task planning → task scheduling.
"""

import sys
import os
import pytest
from pathlib import Path

# Add the agents directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import run_full_study_workflow


@pytest.mark.integration
class TestFullStudyWorkflow:
    """End-to-end tests for the complete study workflow."""

    def test_complete_workflow_from_pdf_to_schedule(self):
        """Test the complete workflow from PDF to scheduled study sessions."""
        # Path to the test PDF
        test_pdf_path = os.path.join(
            os.path.dirname(__file__), "..", "course_ingestion", "tests", "test.pdf"
        )

        # Skip test if PDF doesn't exist
        if not os.path.exists(test_pdf_path):
            pytest.skip(f"Test PDF not found at: {test_pdf_path}")

        # Test parameters
        learning_goal = "Learn Big Data concepts and Hadoop ecosystem"
        available_time = 300  # 5 hours total study time

        print("🚀 Starting complete study workflow test...")
        print(f"📖 PDF: {test_pdf_path}")
        print(f"🎯 Goal: {learning_goal}")
        print(f"⏰ Available time: {available_time} minutes")

        # Run the complete workflow
        result = run_full_study_workflow(
            pdf_paths=[test_pdf_path],
            learning_goal=learning_goal,
            available_time=available_time,
        )

        # Verify the result structure
        assert "planner_output" in result
        assert "scheduler_output" in result
        assert "metadata" in result

        # Verify planner output
        planner_output = result["planner_output"]
        assert "task_graph" in planner_output
        assert "study_plan_id" in planner_output
        assert "course_id" in planner_output

        task_graph = planner_output["task_graph"]
        assert "goal" in task_graph
        assert "tasks" in task_graph
        assert isinstance(task_graph["tasks"], list)
        assert len(task_graph["tasks"]) > 0, "Planner should generate at least one task"

        # Verify scheduler output
        scheduler_output = result["scheduler_output"]
        assert "sessions" in scheduler_output
        assert "total_minutes" in scheduler_output
        assert "fallback_used" in scheduler_output
        assert "skipped_tasks" in scheduler_output
        assert "span_days" in scheduler_output

        sessions = scheduler_output["sessions"]
        assert isinstance(sessions, list)
        assert len(sessions) > 0, "Scheduler should create at least one session"

        total_minutes = scheduler_output["total_minutes"]
        assert total_minutes > 0, "Study plan should have positive total minutes"

        # Verify metadata
        metadata = result["metadata"]
        assert "course_id" in metadata
        assert "study_plan_id" in metadata
        assert "total_tasks" in metadata
        assert "scheduled_sessions" in metadata
        assert "total_scheduled_minutes" in metadata
        assert "plan_span_days" in metadata

        # Verify data consistency
        assert metadata["total_tasks"] == len(task_graph["tasks"])
        assert metadata["scheduled_sessions"] == len(sessions)
        assert metadata["total_scheduled_minutes"] == total_minutes

        # Verify session structure
        for i, session in enumerate(sessions):
            print(f"Session {i}: {session}")
            assert "task_id" in session
            assert "start_datetime" in session
            assert "end_datetime" in session
            assert "slot_score" in session
            assert "scheduled" in session

            # Verify datetime objects are valid (they should be datetime objects, not strings)
            from datetime import datetime
            start_dt = session["start_datetime"]
            end_dt = session["end_datetime"]
            assert isinstance(start_dt, datetime), f"start_datetime should be datetime object, got {type(start_dt)}"
            assert isinstance(end_dt, datetime), f"end_datetime should be datetime object, got {type(end_dt)}"
            assert start_dt < end_dt, "Session end time should be after start time"

        print("✅ Complete workflow test passed!")
        print(f"📊 Results:")
        print(f"   • Course ID: {metadata['course_id']}")
        print(f"   • Study Plan ID: {metadata['study_plan_id']}")
        print(f"   • Tasks generated: {metadata['total_tasks']}")
        print(f"   • Sessions scheduled: {metadata['scheduled_sessions']}")
        print(f"   • Total study time: {metadata['total_scheduled_minutes']} minutes")
        print(f"   • Plan spans: {metadata['plan_span_days']} days")
        print(f"   • Fallback used: {scheduler_output['fallback_used']}")
        print(f"   • Tasks skipped: {len(scheduler_output['skipped_tasks'])}")

    def test_workflow_with_invalid_pdf(self):
        """Test that the workflow handles invalid PDF paths gracefully."""
        with pytest.raises((FileNotFoundError, ValueError)):
            run_full_study_workflow(
                pdf_paths=["/nonexistent/file.pdf"],
                learning_goal="Test goal",
                available_time=60,
            )

    def test_workflow_with_empty_goal(self):
        """Test that the workflow handles empty goals appropriately."""
        test_pdf_path = os.path.join(
            os.path.dirname(__file__), "..", "course_ingestion", "tests", "test.pdf"
        )

        if not os.path.exists(test_pdf_path):
            pytest.skip(f"Test PDF not found at: {test_pdf_path}")

        result = run_full_study_workflow(
            pdf_paths=[test_pdf_path],
            learning_goal="",  # Empty goal
            available_time=60,
        )

        # Should still complete but might have clarification warnings
        assert "planner_output" in result
        assert "scheduler_output" in result
        assert "metadata" in result


if __name__ == "__main__":
    # Run the tests
    test_instance = TestFullStudyWorkflow()

    print("Running end-to-end workflow test...")
    try:
        test_instance.test_complete_workflow_from_pdf_to_schedule()
        print("✅ End-to-end workflow test passed!")
    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {e}")
        raise