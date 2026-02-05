#!/usr/bin/env python3
"""
Test script to demonstrate the study partner system with goal storage.

This script shows how to:
1. Process a PDF course material
2. Generate a study plan
3. Store the study plan in the database
4. Retrieve and display stored study plans
"""

import sys
import os
from pathlib import Path

# Add the agents directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import run_study_planner, run_study_planner_with_course_id
from agents.course_ingestion.services.database_service import DatabaseService


def test_with_pdf():
    """Test the system with a PDF file and store goals in database."""
    print("🧪 Testing Study Partner System with PDF")
    print("=" * 50)

    # Path to test PDF
    test_pdf_path = os.path.join("course_ingestion", "tests", "test.pdf")

    if not os.path.exists(test_pdf_path):
        print(f"❌ Test PDF not found at: {test_pdf_path}")
        return

    print(f"📄 Using test PDF: {test_pdf_path}")

    # Define learning goal and time
    learning_goal = "Learn Big Data concepts and Hadoop ecosystem"
    available_time = 120  # 2 hours

    print(f"🎯 Goal: {learning_goal}")
    print(f"⏰ Available time: {available_time} minutes")
    print()

    # Run the study planner
    result = run_study_planner(
        pdf_paths=[test_pdf_path],
        learning_goal=learning_goal,
        available_time=available_time,
    )

    # Display results
    print("✅ Study Plan Generated!")
    print(f"📚 Course ID: {result['course_id']}")
    print(f"📋 Study Plan ID: {result['study_plan_id']}")
    print(f"🎯 Goal: {result['task_graph']['goal']}")
    print(
        f"📊 Total estimated time: {result['task_graph']['total_estimated_minutes']} minutes"
    )
    print(f"📝 Number of tasks: {len(result['task_graph']['tasks'])}")
    print()

    # Show tasks
    print("📋 Generated Tasks:")
    for i, task in enumerate(result["task_graph"]["tasks"], 1):
        print(f"  {i}. {task['title']}")
        print(
            f"     ⏱️  {task['estimated_minutes']} min | 💪 Difficulty: {task['difficulty']}"
        )
        if task["prerequisites"]:
            print(f"     🔗 Prerequisites: {', '.join(task['prerequisites'])}")
        print()

    return result


def demonstrate_database_storage():
    """Demonstrate retrieving stored study plans from database."""
    print("🗄️  Database Storage Demonstration")
    print("=" * 50)

    db = DatabaseService()

    # Get all study plans (in a real app, you'd filter by user)
    study_plans = list(db.study_plan_collection.find().limit(5))

    print(f"📊 Found {len(study_plans)} study plans in database")
    print()

    for i, plan in enumerate(study_plans, 1):
        print(f"Study Plan {i}:")
        print(f"  🆔 ID: {plan['_id']}")
        print(f"  📚 Course ID: {plan.get('course_id', 'N/A')}")
        print(f"  🎯 Goal: {plan['task_graph']['goal']}")
        print(f"  📝 Tasks: {len(plan['task_graph']['tasks'])}")
        print(f"  ⏰ Total time: {plan['task_graph']['total_estimated_minutes']} min")
        print(f"  📅 Created: {plan.get('created_at', 'N/A')}")
        print()


def test_with_existing_course():
    """Test generating a study plan using an existing course ID."""
    print("🔄 Testing with Existing Course ID")
    print("=" * 50)

    db = DatabaseService()

    # Get the most recent course
    recent_course = db.collection.find_one(sort=[("_id", -1)])
    if not recent_course:
        print("❌ No courses found in database")
        return

    course_id = str(recent_course["_id"])
    print(f"📚 Using existing course ID: {course_id}")

    # Generate a new study plan for the same course
    new_goal = "Master Hadoop MapReduce programming patterns"
    available_time = 90

    result = run_study_planner_with_course_id(
        course_id=course_id, learning_goal=new_goal, available_time=available_time
    )

    print("✅ New study plan generated!")
    print(f"📋 Study Plan ID: {result['study_plan_id']}")
    print(f"🎯 Goal: {result['task_graph']['goal']}")
    print(f"📝 Tasks: {len(result['task_graph']['tasks'])}")
    print()


if __name__ == "__main__":
    try:
        # Test with PDF
        pdf_result = test_with_pdf()

        # Demonstrate database storage
        demonstrate_database_storage()

        # Test with existing course
        test_with_existing_course()

        print("🎉 All tests completed successfully!")
        print("💡 Goals are now stored in the database and can be retrieved later.")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
