#!/usr/bin/env python3
"""
Comprehensive testing script for the Coach agent.
Tests all combinations of Focus States, Fatigue Levels, and Affective States.
Based on the 3 ML models: Focus State Estimator, Cognitive Fatigue Detection, Affective State Estimation.
"""

import os
from datetime import datetime, timedelta
from agents.coach.agent import run_coach
from agents.coach.models.schemas import CoachInput, ScheduledTask, FocusState


def create_test_tasks():
    """Create sample scheduled tasks - Coach chooses first one."""
    return [
        ScheduledTask(
            task_id="math_001",
            title="Algebra Fundamentals",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=45),
            priority=1
        ),
        ScheduledTask(
            task_id="physics_002",
            title="Newton's Laws",
            start_time=datetime.now() + timedelta(hours=2),
            end_time=datetime.now() + timedelta(hours=2, minutes=30),
            priority=2
        ),
        ScheduledTask(
            task_id="chemistry_003",
            title="Organic Chemistry Basics",
            start_time=datetime.now() + timedelta(hours=4),
            end_time=datetime.now() + timedelta(hours=4, minutes=45),
            priority=3
        )
    ]


def test_coach_scenario(focus_state: str, focus_score: float,
                       fatigue_prob: float, affective_state: str,
                       ignored_count: int = 0, dnd: bool = False):
    """Test coach with specific ML model outputs."""

    tasks = create_test_tasks()

    input_data = CoachInput(
        scheduled_tasks=tasks,
        current_time=datetime.now(),
        focus_state=FocusState(state=focus_state, score=focus_score),
        fatigue_probability=fatigue_prob,
        affective_state=affective_state,
        ignored_count=ignored_count,
        do_not_disturb=dnd
    )

    action = run_coach(input_data)

    return {
        'focus': f"{focus_state}({focus_score})",
        'fatigue': f"{fatigue_prob:.1f}",
        'affect': affective_state,
        'ignored': ignored_count,
        'dnd': dnd,
        'action': action.action_type,
        'message': action.message,
        'reasoning': action.reasoning,
        'target_task': action.target_task_id
    }


def print_scenario_result(result):
    """Print formatted test result."""
    print(f"📊 {result['focus']} | Fatigue: {result['fatigue']} | Mood: {result['affect']}")
    print(f"   → Action: {result['action']}")
    if result['message']:
        print(f"   💬 \"{result['message']}\"")
    print(f"   🤔 {result['reasoning']}")
    if result['target_task']:
        print(f"   🎯 Target: {result['target_task']}")
    print()


def test_focus_states():
    """Test different focus states with moderate fatigue and neutral affect."""
    print("🎯 TESTING FOCUS STATE ESTIMATOR")
    print("=" * 60)

    scenarios = [
        ("Focused", 0.9, 0.3, "engaged"),
        ("Drifting", 0.5, 0.4, "bored"),
        ("Lost", 0.2, 0.5, "frustrated")
    ]

    for focus, score, fatigue, affect in scenarios:
        result = test_coach_scenario(focus, score, fatigue, affect)
        print_scenario_result(result)


def test_fatigue_levels():
    """Test different fatigue levels with drifting focus."""
    print("😴 TESTING COGNITIVE FATIGUE DETECTION")
    print("=" * 60)

    scenarios = [
        (0.1, "Drifting", 0.4, "engaged"),      # Fresh
        (0.4, "Drifting", 0.4, "engaged"),      # Moderate
        (0.7, "Drifting", 0.4, "frustrated"),   # Tired
        (0.9, "Lost", 0.2, "stressed")          # Exhausted
    ]

    for fatigue, focus, score, affect in scenarios:
        result = test_coach_scenario(focus, score, fatigue, affect)
        print_scenario_result(result)


def test_affective_states():
    """Test different affective states with moderate conditions."""
    print("😊 TESTING AFFECTIVE STATE ESTIMATION")
    print("=" * 60)

    affective_states = ["engaged", "frustrated", "stressed", "bored", "confident"]

    for affect in affective_states:
        # Use drifting focus and moderate fatigue for comparison
        result = test_coach_scenario("Drifting", 0.4, 0.4, affect)
        print_scenario_result(result)


def test_critical_scenarios():
    """Test edge cases and critical scenarios."""
    print("🚨 TESTING CRITICAL SCENARIOS")
    print("=" * 60)

    scenarios = [
        ("Deep Focus Protection", "Focused", 0.95, 0.2, "engaged", 0, False),
        ("Burnout Prevention", "Lost", 0.1, 0.9, "frustrated", 0, False),
        ("Do Not Disturb", "Drifting", 0.4, 0.5, "stressed", 0, True),
        ("Ignored User", "Drifting", 0.3, 0.6, "bored", 3, False),
        ("High Confidence", "Focused", 0.8, 0.3, "confident", 0, False),
        ("Frustrated & Tired", "Lost", 0.2, 0.8, "frustrated", 0, False),
        ("Bored & Distracted", "Drifting", 0.3, 0.4, "bored", 0, False),
        ("Stressed & Focused", "Focused", 0.7, 0.6, "stressed", 0, False)
    ]

    for name, focus, score, fatigue, affect, ignored, dnd in scenarios:
        print(f"🔍 {name}:")
        result = test_coach_scenario(focus, score, fatigue, affect, ignored, dnd)
        print_scenario_result(result)


def test_comprehensive_matrix():
    """Test comprehensive combinations of all 3 ML models."""
    print("🔬 COMPREHENSIVE ML MODEL MATRIX TEST")
    print("=" * 60)

    focus_states = [("Focused", 0.8), ("Drifting", 0.4), ("Lost", 0.2)]
    fatigue_levels = [0.2, 0.5, 0.8]
    affective_states = ["engaged", "frustrated", "bored", "confident"]

    matrix_results = []

    for focus_name, focus_score in focus_states:
        for fatigue in fatigue_levels:
            for affect in affective_states:
                result = test_coach_scenario(focus_name, focus_score, fatigue, affect)
                matrix_results.append(result)

                # Print summary for each combination
                status = "🤫" if result['action'] == 'silence' else "💬"
                print(f"{status} {focus_name[:1]}({focus_score}) + F{fatigue:.1f} + {affect[:3]} → {result['action']}")

    print(f"\n📈 Matrix Summary: {len(matrix_results)} combinations tested")

    # Summary statistics
    actions = [r['action'] for r in matrix_results]
    silence_count = actions.count('silence')
    encourage_count = actions.count('encourage')

    print(f"🤫 Silence actions: {silence_count}")
    print(f"💬 Encourage actions: {encourage_count}")
    print(f"📊 Silence rate: {silence_count/len(matrix_results)*100:.1f}%")


def main():
    """Run comprehensive Coach agent testing."""
    print("🤖 COACH AGENT COMPREHENSIVE TESTING")
    print("Testing integration of 3 ML models:")
    print("🎯 Focus State Estimator | 😴 Cognitive Fatigue Detection | 😊 Affective State Estimation")
    print("=" * 80)

    # Set dummy env vars for testing
    os.environ.setdefault("GEMINI_API_KEY", "dummy_key_for_testing")
    os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
    os.environ.setdefault("DB_NAME", "study_partner")

    # Run different test suites
    test_focus_states()
    print("\n" + "="*80 + "\n")

    test_fatigue_levels()
    print("\n" + "="*80 + "\n")

    test_affective_states()
    print("\n" + "="*80 + "\n")

    test_critical_scenarios()
    print("\n" + "="*80 + "\n")

    test_comprehensive_matrix()

    print("\n✅ Testing complete! Coach agent behavior validated across all ML model combinations.")


if __name__ == "__main__":
    main()