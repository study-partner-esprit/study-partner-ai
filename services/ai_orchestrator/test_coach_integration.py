"""Test script for coach integration with fatigue detection.

This script tests the complete flow:
1. Initialize AIOrchestrator
2. Create a signal snapshot with fatigue data
3. Run the coach and verify it uses fatigue information
4. Test rule engine with different fatigue states
"""

import sys
from pathlib import Path
from datetime import datetime

# Add root to path
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))

from services.ai_orchestrator.orchestrator import AIOrchestrator
from services.signal_processing_service.service import SignalProcessingService
from agents.coach.models.schemas import CoachInput, FocusState, FatigueState, ScheduledTask


def test_coach_fatigue_integration():
    """Test the integrated coach with fatigue detection."""
    print("=" * 60)
    print("Testing Coach Integration with Fatigue Detection")
    print("=" * 60)

    # Initialize orchestrator
    print("\n1. Initializing AIOrchestrator...")
    orchestrator = AIOrchestrator()

    # Test with different fatigue states
    test_cases = [
        ("Alert", 0.15, "Should allow normal coaching"),
        ("Moderate", 0.45, "Should be neutral"),
        ("High", 0.75, "Should suggest break"),
        ("Critical", 0.95, "Should force break")
    ]

    for fatigue_state, fatigue_score, expected in test_cases:
        print(f"\n2. Testing fatigue state: {fatigue_state} (score: {fatigue_score})")
        print(f"   Expected: {expected}")

        # Create mock signal snapshot with specific fatigue state
        signal_service = SignalProcessingService()
        snapshot = signal_service.get_current_signal_snapshot(
            user_id="test_user_fatigue",
            video_features=None,  # Mock focus data
            video_frame=None      # Mock fatigue data
        )

        # Override the fatigue data in the snapshot
        snapshot.fatigue_state = fatigue_state
        snapshot.fatigue_score = fatigue_score
        # Set focus to a high state to test fatigue override
        snapshot.focus_state = "Focused"
        snapshot.focus_score = 0.85
        snapshot.focus_confidence = 0.90
        # Ensure unique timestamp
        from datetime import datetime
        snapshot.timestamp = datetime.now()

        # Manually save this snapshot so the orchestrator can find it
        signal_service.repository.save_signal_snapshot(snapshot)

        # Run the coach
        coach_action = orchestrator.run_coach(
            user_id="test_user_fatigue",
            current_time=datetime.now(),
            ignored_count=0,
            do_not_disturb=False
        )

        print(f"   Coach action: {coach_action.action_type}")
        if coach_action.message:
            print(f"   Message: {coach_action.message}")
        print(f"   Reasoning: {coach_action.reasoning}")

        # Verify expected behavior
        if fatigue_state == "Critical":
            assert coach_action.action_type == "suggest_break", f"Expected suggest_break for {fatigue_state}, got {coach_action.action_type}"
            assert "fatigue" in coach_action.reasoning.lower(), f"Expected fatigue mention in reasoning: {coach_action.reasoning}"
        elif fatigue_state == "High":
            # High fatigue should suggest break unless deeply focused
            # Since focus is high, it should still be silent (focus overrides high fatigue)
            assert coach_action.action_type == "silence", f"Expected silence for high focus overriding {fatigue_state}: {coach_action.action_type}"
        else:
            # Alert/Moderate fatigue with high focus should be silent
            assert coach_action.action_type == "silence", f"Expected silence for {fatigue_state} with high focus: {coach_action.action_type}"

    print("\n" + "=" * 60)
    print("✓ Coach fatigue integration test complete!")
    print("=" * 60)


def test_rule_engine_directly():
    """Test the rule engine directly with different fatigue states."""
    print("\n" + "=" * 60)
    print("Testing Rule Engine Directly")
    print("=" * 60)

    from agents.coach.rules.rule_engine import apply_rules

    # Test critical fatigue rule
    print("\n1. Testing critical fatigue rule...")
    critical_input = CoachInput(
        scheduled_tasks=[],
        current_time=datetime.now(),
        focus_state=FocusState(state="Drifting", score=0.3),
        fatigue_state=FatigueState(state="Critical", score=0.95),
        affective_state="stressed",
        ignored_count=0,
        do_not_disturb=False,
        is_late=False
    )

    action = apply_rules(critical_input)
    assert action is not None, "Critical fatigue should trigger a rule"
    assert action.action_type == "suggest_break", f"Expected suggest_break, got {action.action_type}"
    print("   ✓ Critical fatigue correctly triggers break suggestion")

    # Test high fatigue rule
    print("\n2. Testing high fatigue rule...")
    high_input = CoachInput(
        scheduled_tasks=[],
        current_time=datetime.now(),
        focus_state=FocusState(state="Drifting", score=0.4),
        fatigue_state=FatigueState(state="High", score=0.75),
        affective_state="engaged",
        ignored_count=0,
        do_not_disturb=False,
        is_late=False
    )

    action = apply_rules(high_input)
    assert action is not None, "High fatigue should trigger a rule"
    assert action.action_type == "suggest_break", f"Expected suggest_break, got {action.action_type}"
    print("   ✓ High fatigue correctly triggers break suggestion")

    # Test focus override
    print("\n3. Testing focus override for high fatigue...")
    focused_input = CoachInput(
        scheduled_tasks=[],
        current_time=datetime.now(),
        focus_state=FocusState(state="Focused", score=0.8),
        fatigue_state=FatigueState(state="High", score=0.75),
        affective_state="engaged",
        ignored_count=0,
        do_not_disturb=False,
        is_late=False
    )

    action = apply_rules(focused_input)
    assert action is not None, "Deep focus should still trigger silence rule"
    assert action.action_type == "silence", f"Expected silence for deep focus, got {action.action_type}"
    print("   ✓ Deep focus correctly overrides high fatigue (prioritizes focus)")

    print("\n" + "=" * 60)
    print("✓ Rule engine tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_coach_fatigue_integration()
    test_rule_engine_directly()
