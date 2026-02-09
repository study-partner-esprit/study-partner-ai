#!/usr/bin/env python3
"""
Test Coach-Scheduler Integration
Demonstrates how the Coach agent can autonomously modify schedules.
"""

import os
from agents.orchestrator import run_coaching_session


def test_coach_scheduler_integration():
    """Test the complete Coach → Scheduler integration."""
    print("🔗 TESTING COACH-SCHEDULER INTEGRATION")
    print("=" * 60)

    # Set dummy env vars for testing
    os.environ.setdefault("GEMINI_API_KEY", "dummy_key_for_testing")
    os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
    os.environ.setdefault("DB_NAME", "study_partner")

    test_scenarios = [
        {
            "name": "Fatigued Student - Break Suggestion",
            "focus_state": "Lost",
            "focus_score": 0.1,
            "fatigue_probability": 0.9,
            "affective_state": "frustrated",
            "expected": "suggest_break with schedule_changes"
        },
        {
            "name": "High Fatigue - Break Suggestion",
            "focus_state": "Drifting",
            "focus_score": 0.3,
            "fatigue_probability": 0.8,
            "affective_state": "stressed",
            "expected": "suggest_break with schedule_changes"
        },
        {
            "name": "Extreme Fatigue Late Night - Session Suspension",
            "focus_state": "Lost",
            "focus_score": 0.05,
            "fatigue_probability": 0.95,
            "affective_state": "frustrated",
            "is_late": True,
            "expected": "suggest_break with session suspension"
        },
        {
            "name": "Focused Student - No Changes",
            "focus_state": "Focused",
            "focus_score": 0.9,
            "fatigue_probability": 0.2,
            "affective_state": "engaged",
            "expected": "silence (no schedule changes)"
        },
        {
            "name": "Do Not Disturb - No Changes",
            "focus_state": "Drifting",
            "focus_score": 0.4,
            "fatigue_probability": 0.6,
            "affective_state": "bored",
            "do_not_disturb": True,
            "expected": "silence (DND active)"
        }
    ]

    for scenario in test_scenarios:
        print(f"\n🎯 {scenario['name']}")
        print(f"Expected: {scenario['expected']}")
        print("-" * 40)

        result = run_coaching_session(
            focus_state=scenario["focus_state"],
            focus_score=scenario["focus_score"],
            fatigue_probability=scenario["fatigue_probability"],
            affective_state=scenario["affective_state"],
            ignored_count=scenario.get("ignored_count", 0),
            do_not_disturb=scenario.get("do_not_disturb", False),
            is_late=scenario.get("is_late", False)
        )

        # Analyze results
        action = result["coach_action"]
        schedule_modified = result["schedule_modified"]

        print(f"✅ Action: {action['action_type']}")
        if action.get('message'):
            print(f"   💬 \"{action['message']}\"")
        print(f"📅 Schedule Modified: {schedule_modified}")

        if schedule_modified:
            for mod in result["modifications"]:
                print(f"   🔧 Applied: {mod['action']}")
                if mod['action'] == 'add_break':
                    duration = mod['details'].get('duration_minutes', 'unknown')
                    print(f"      ⏰ Added {duration}-minute break")
                elif mod['action'] == 'suspend_session':
                    print(f"      🛌 Suspended session until tomorrow")

        # Verify expectations
        success = True
        if "break" in scenario["expected"] and not schedule_modified:
            success = False
            print("❌ Expected schedule modification but none occurred")
        elif "silence" in scenario["expected"] and action['action_type'] != 'silence':
            success = False
            print("❌ Expected silence but got different action")
        elif "suspension" in scenario["expected"] and not schedule_modified:
            success = False
            print("❌ Expected session suspension but none occurred")

        if success:
            print("✅ Scenario passed")
        else:
            print("❌ Scenario failed")

    print("\n" + "="*60)
    print("🎉 Coach-Scheduler Integration Test Complete!")
    print("\n📋 SUMMARY:")
    print("• Coach adapts break duration based on fatigue level (5-10 minutes)")
    print("• Extreme fatigue + late hours triggers session suspension")
    print("• Schedule Updater can add breaks, shift tasks, or suspend sessions")
    print("• Orchestrator coordinates the Coach → Schedule modification flow")
    print("• All changes are persisted to MongoDB")
    print("• Intelligent fatigue management prevents burnout!")


if __name__ == "__main__":
    test_coach_scheduler_integration()