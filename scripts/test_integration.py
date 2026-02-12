#!/usr/bin/env python3
"""
Quick integration test for AI services.
Tests the API endpoints without needing the frontend.
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"
TEST_USER_ID = "test_user_integration"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_health():
    """Test health check endpoint."""
    print_section("Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        print(f"✓ API is healthy")
        print(f"  Services: {json.dumps(data['services'], indent=2)}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_signals():
    """Test signal processing."""
    print_section("Signal Processing")
    try:
        response = requests.get(f"{BASE_URL}/api/ai/signals/current/{TEST_USER_ID}")
        data = response.json()
        print(f"✓ Signals fetched successfully")
        print(f"  Focus: {data['focus']['state']} ({data['focus']['score']:.2f})")
        print(f"  Fatigue: {data['fatigue']['state']} ({data['fatigue']['score']:.2f})")
        return True
    except Exception as e:
        print(f"✗ Signal processing failed: {e}")
        return False

def test_coach():
    """Test coach decision."""
    print_section("Coach Decision")
    try:
        response = requests.post(
            f"{BASE_URL}/api/ai/coach/decision",
            json={
                "user_id": TEST_USER_ID,
                "ignored_count": 0,
                "do_not_disturb": False
            }
        )
        data = response.json()
        coach_action = data['coach_action']
        print(f"✓ Coach decision received")
        print(f"  Action: {coach_action['action_type']}")
        print(f"  Reasoning: {coach_action['reasoning'][:100]}...")
        if coach_action.get('message'):
            print(f"  Message: {coach_action['message'][:100]}...")
        return True
    except Exception as e:
        print(f"✗ Coach decision failed: {e}")
        return False

def test_planner():
    """Test study planner."""
    print_section("Study Planner")
    try:
        response = requests.post(
            f"{BASE_URL}/api/ai/planner/create-plan",
            json={
                "user_id": TEST_USER_ID,
                "goal": "Learn basic Python programming",
                "available_time_minutes": 90,
                "start_date": datetime.now().isoformat()
            },
            timeout=30
        )
        data = response.json()
        plan = data['plan']
        print(f"✓ Study plan created")
        print(f"  Schedule ID: {data['schedule_id']}")
        print(f"  Tasks: {len(plan['task_graph']['atomic_tasks'])}")
        print(f"  Total time: {plan['task_graph']['total_estimated_minutes']} minutes")
        
        # Print first few tasks
        for i, task in enumerate(plan['task_graph']['atomic_tasks'][:3]):
            print(f"  {i+1}. {task['title']} ({task['estimated_minutes']}min)")
        
        return True
    except Exception as e:
        print(f"✗ Study planner failed: {e}")
        return False

def main():
    print("\n🧪 Study Partner AI - Integration Test")
    print("="*60)
    print(f"Testing API at: {BASE_URL}")
    print(f"Test User ID: {TEST_USER_ID}")
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health()))
    results.append(("Signal Processing", test_signals()))
    results.append(("Coach Decision", test_coach()))
    results.append(("Study Planner", test_planner()))
    
    # Summary
    print_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! AI integration is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
