"""
Test script to demonstrate course-to-plan workflow.

This shows how the planner now:
1. Takes a course from ingestion
2. Generates tasks from the course structure
3. Creates a study plan with proper scheduling
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from agents.orchestrator import run_study_planner

def test_course_to_plan():
    """Test generating a plan directly from course content."""
    
    # Path to test PDF
    test_pdf = "agents/course_ingestion/tests/test.pdf"
    
    if not Path(test_pdf).exists():
        print(f"❌ Test PDF not found at: {test_pdf}")
        return
    
    print("=" * 60)
    print("Testing Course-to-Plan Workflow")
    print("=" * 60)
    
    # Test 1: Generate plan without specific goal (derive from course)
    print("\n📋 Test 1: Generate plan from course structure (no specific goal)")
    print("-" * 60)
    
    result = run_study_planner(
        pdf_paths=[test_pdf],
        learning_goal=None,  # No goal - will be derived from course
        available_time=480  # 8 hours
    )
    
    print(f"\n✅ Plan generated successfully!")
    print(f"   Course ID: {result['course_id']}")
    print(f"   Study Plan ID: {result['study_plan_id']}")
    
    task_graph = result['task_graph']
    print(f"\n📊 Plan Summary:")
    print(f"   Goal: {task_graph['goal']}")
    print(f"   Total tasks: {len(task_graph['tasks'])}")
    print(f"   Total estimated time: {task_graph['total_estimated_minutes']} minutes")
    
    if task_graph['tasks']:
        print(f"\n📝 Sample Tasks:")
        for i, task in enumerate(task_graph['tasks'][:5]):
            print(f"   {i+1}. {task['title']}")
            print(f"      Duration: {task['estimated_minutes']} min | Difficulty: {task['difficulty']}")
    
    # Test 2: Generate plan with specific goal (focuses on goal using course content)
    print("\n\n📋 Test 2: Generate plan with specific goal + course content")
    print("-" * 60)
    
    result = run_study_planner(
        pdf_paths=[test_pdf],
        learning_goal="Learn Hadoop and MapReduce programming",
        available_time=240  # 4 hours
    )
    
    print(f"\n✅ Plan generated successfully!")
    task_graph = result['task_graph']
    print(f"\n📊 Plan Summary:")
    print(f"   Goal: {task_graph['goal']}")
    print(f"   Total tasks: {len(task_graph['tasks'])}")
    print(f"   Total estimated time: {task_graph['total_estimated_minutes']} minutes")
    
    if task_graph['tasks']:
        print(f"\n📝 Sample Tasks:")
        for i, task in enumerate(task_graph['tasks'][:5]):
            print(f"   {i+1}. {task['title']}")
            print(f"      Duration: {task['estimated_minutes']} min | Difficulty: {task['difficulty']}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_course_to_plan()
