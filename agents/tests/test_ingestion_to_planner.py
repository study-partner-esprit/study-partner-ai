"""
Integration tests for the ingestion-to-planner pipeline.

Tests the complete flow from PDF ingestion to study plan generation.
"""

import sys
import os
import pytest
from pathlib import Path

# Add the agents directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import run_study_planner


class TestIngestionToPlannerIntegration:
    """Integration tests for the complete ingestion-to-planning pipeline."""

    def test_full_pipeline_with_test_pdf(self):
        """Test the complete pipeline using the existing test PDF."""
        # Path to the test PDF
        test_pdf_path = os.path.join(
            os.path.dirname(__file__), "..", "course_ingestion", "tests", "test.pdf"
        )

        # Skip test if PDF doesn't exist
        if not os.path.exists(test_pdf_path):
            pytest.skip(f"Test PDF not found at: {test_pdf_path}")

        # Test parameters
        learning_goal = "Learn Big Data concepts and Hadoop ecosystem"
        available_time = 120  # 2 hours

        # Run the complete pipeline
        result = run_study_planner(
            pdf_paths=[test_pdf_path],
            learning_goal=learning_goal,
            available_time=available_time,
        )

        # Verify the result structure
        assert "task_graph" in result
        assert "warning" in result
        assert "clarification_required" in result

        # Verify task graph structure
        task_graph = result["task_graph"]
        assert "goal" in task_graph
        assert "tasks" in task_graph
        assert "total_estimated_minutes" in task_graph

        # Verify we have at least one task
        tasks = task_graph["tasks"]
        assert len(tasks) > 0, "Planner should generate at least one task"

        # Verify each task has required fields
        for task in tasks:
            assert "id" in task, "Task must have an ID"
            assert "title" in task, "Task must have a title"
            assert "description" in task, "Task must have a description"
            assert "estimated_minutes" in task, "Task must have estimated duration"
            assert "difficulty" in task, "Task must have difficulty level"

            # Verify data types and ranges
            assert isinstance(task["title"], str), "Title must be a string"
            assert isinstance(task["description"], str), "Description must be a string"
            assert isinstance(
                task["estimated_minutes"], int
            ), "Estimated minutes must be an integer"
            assert (
                5 <= task["estimated_minutes"] <= 45
            ), "Estimated minutes must be between 5-45"
            assert isinstance(
                task["difficulty"], (int, float)
            ), "Difficulty must be a number"
            assert (
                0.0 <= task["difficulty"] <= 1.0
            ), "Difficulty must be between 0.0-1.0"

        # Verify the goal matches what we requested
        assert task_graph["goal"] == learning_goal

        print(f"✅ Generated {len(tasks)} tasks for goal: {learning_goal}")
        print(
            f"📊 Total estimated time: {task_graph['total_estimated_minutes']} minutes"
        )

        # Print first few tasks for verification
        for i, task in enumerate(tasks[:3]):
            print(
                f"Task {i+1}: {task['title']} ({task['estimated_minutes']} min, difficulty: {task['difficulty']})"
            )

    def test_pipeline_with_invalid_pdf(self):
        """Test that the pipeline handles invalid PDF paths gracefully."""
        with pytest.raises((FileNotFoundError, ValueError)):
            run_study_planner(
                pdf_paths=["/nonexistent/file.pdf"],
                learning_goal="Test goal",
                available_time=60,
            )

    def test_pipeline_with_empty_goal(self):
        """Test that the pipeline handles empty goals appropriately."""
        test_pdf_path = os.path.join(
            os.path.dirname(__file__), "course_ingestion", "tests", "testing.pdf"
        )

        if not os.path.exists(test_pdf_path):
            pytest.skip(f"Test PDF not found at: {test_pdf_path}")

        result = run_study_planner(
            pdf_paths=[test_pdf_path], learning_goal="", available_time=60  # Empty goal
        )

        # Should require clarification for empty goal
        assert result["clarification_required"] is True
        assert "vague" in (result.get("warning") or "").lower()


if __name__ == "__main__":
    # Run the tests
    test_instance = TestIngestionToPlannerIntegration()

    print("Running integration test...")
    try:
        test_instance.test_full_pipeline_with_test_pdf()
        print("✅ Integration test passed!")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise
