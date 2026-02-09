#!/usr/bin/env python3
"""
Demonstration of Intelligent vs Hardcoded Decision Making
"""

import json

def old_hardcoded_response(fatigue_prob: float, focus_state: str, affective_state: str) -> str:
    """Old hardcoded approach - rigid if-else logic"""
    # Hardcoded logic - only handles specific combinations
    if focus_state == "Lost" and fatigue_prob > 0.7:
        return json.dumps({
            "action": "break",
            "message": "You seem quite tired. How about taking a 5-minute break to recharge?",
            "reasoning": "High fatigue levels combined with lost focus indicate need for rest."
        })
    else:
        return json.dumps({
            "action": "continue",
            "message": "You're doing great! Keep up the good work.",
            "reasoning": "Student shows good engagement."
        })

def new_intelligent_response(fatigue_prob: float, focus_state: str, focus_score: float,
                           affective_state: str, is_late: bool) -> str:
    """New intelligent approach - multi-factor analysis with severity scoring"""

    # Intelligent severity scoring (0-10 scale)
    severity_score = 0

    # Focus issues (0-3 points)
    if focus_state == "Lost":
        severity_score += 3
    elif focus_state == "Drifting":
        severity_score += 2 if focus_score < 0.4 else 1

    # Fatigue impact (0-3 points)
    if fatigue_prob > 0.8:
        severity_score += 3
    elif fatigue_prob > 0.6:
        severity_score += 2

    # Emotional state impact (0-2 points)
    if affective_state in ["frustrated", "stressed"]:
        severity_score += 2
    elif affective_state == "bored":
        severity_score += 1

    # Time factor bonus (0-2 points)
    if is_late and fatigue_prob > 0.7:
        severity_score += 2

    # Deep focus protection - never interrupt high focus
    if focus_state == "Focused" and focus_score > 0.8:
        return json.dumps({
            "action": "silence",
            "message": None,
            "reasoning": f"Exceptional focus (score: {focus_score:.1f}) in flow state. Severity: {severity_score}/10."
        })

    # Critical fatigue + late night = session suspension
    if fatigue_prob > 0.9 and is_late:
        return json.dumps({
            "action": "suspend_session",
            "message": f"Fatigue level ({fatigue_prob:.1f}) + late hour requires session suspension.",
            "reasoning": f"Critical threshold exceeded. Severity: {severity_score}/10."
        })

    # High fatigue = break with appropriate duration
    if fatigue_prob > 0.7:
        break_duration = 10 if fatigue_prob > 0.8 else 5
        return json.dumps({
            "action": "add_break",
            "message": f"Fatigue ({fatigue_prob:.1f}) suggests {break_duration}-minute break.",
            "reasoning": f"Strategic intervention needed. Severity: {severity_score}/10."
        })

    # Emotional support for challenging states
    if affective_state == "frustrated" and focus_state == "Lost":
        return json.dumps({
            "action": "encourage",
            "message": "Your persistence despite frustration shows dedication.",
            "reasoning": f"Emotional support prioritized. Severity: {severity_score}/10."
        })

    return json.dumps({
        "action": "encourage",
        "message": "Good progress with balanced engagement.",
        "reasoning": f"Stable state maintained. Severity: {severity_score}/10."
    })

# Test scenarios
test_cases = [
    {
        "name": "High Fatigue + Lost Focus",
        "fatigue_prob": 0.85,
        "focus_state": "Lost",
        "focus_score": 0.2,
        "affective_state": "frustrated",
        "is_late": False
    },
    {
        "name": "Deep Focus Protection",
        "fatigue_prob": 0.3,
        "focus_state": "Focused",
        "focus_score": 0.9,
        "affective_state": "confident",
        "is_late": False
    },
    {
        "name": "Extreme Fatigue Late Night",
        "fatigue_prob": 0.95,
        "focus_state": "Lost",
        "focus_score": 0.1,
        "affective_state": "stressed",
        "is_late": True
    }
]

print("🔍 INTELLIGENT vs HARDCODED DECISION MAKING")
print("=" * 60)

for test_case in test_cases:
    print(f"\n🎯 {test_case['name']}")
    print("-" * 40)

    # Old hardcoded approach
    old_response = json.loads(old_hardcoded_response(
        test_case['fatigue_prob'],
        test_case['focus_state'],
        test_case['affective_state']
    ))
    print(f"🤖 HARDCODED: {old_response['action']} - {old_response['reasoning']}")

    # New intelligent approach
    new_response = json.loads(new_intelligent_response(
        test_case['fatigue_prob'],
        test_case['focus_state'],
        test_case['focus_score'],
        test_case['affective_state'],
        test_case['is_late']
    ))
    print(f"🧠 INTELLIGENT: {new_response['action']} - {new_response['reasoning']}")

print("\n" + "=" * 60)
print("✨ KEY IMPROVEMENTS:")
print("• Multi-factor severity scoring (0-10 scale)")
print("• Structured data parsing instead of string matching")
print("• Nuanced decision-making based on combinations")
print("• Personalized responses with specific metrics")
print("• Context-aware reasoning with quantitative analysis")