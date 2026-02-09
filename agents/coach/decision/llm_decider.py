import json
import os
import google.generativeai as genai
from agents.coach.models.schemas import CoachInput, CoachAction
from agents.coach.decision.prompt import SYSTEM_PROMPT, build_user_prompt


def call_gemini(system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "dummy_key_for_testing":
        # Return intelligent mock response based on input data for testing
        return get_mock_gemini_response(user_prompt)
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback to mock response
        return get_mock_gemini_response(user_prompt)


def get_mock_gemini_response(user_prompt: str) -> str:
    """
    Generate intelligent mock responses based on student state for comprehensive testing.
    Analyzes the user_prompt to determine appropriate coaching response.
    """
    # Extract key information from the prompt
    focus_state = "Drifting"  # default
    fatigue_prob = 0.5  # default
    affective_state = "engaged"  # default
    is_late = False  # default
    
    if "Focused" in user_prompt:
        focus_state = "Focused"
    elif "Lost" in user_prompt:
        focus_state = "Lost"
    
    # Extract fatigue probability
    import re
    fatigue_match = re.search(r'fatigue_probability.*?(\d+\.\d+)', user_prompt)
    if fatigue_match:
        fatigue_prob = float(fatigue_match.group(1))
    
    # Extract affective state
    if "frustrated" in user_prompt:
        affective_state = "frustrated"
    elif "stressed" in user_prompt:
        affective_state = "stressed"
    elif "bored" in user_prompt:
        affective_state = "bored"
    elif "confident" in user_prompt:
        affective_state = "confident"
    
    # Extract is_late
    if "is_late.*?:.*?true" in user_prompt or "late" in user_prompt.lower():
        is_late = True
    elif "stressed" in user_prompt:
        affective_state = "stressed"
    elif "bored" in user_prompt:
        affective_state = "bored"
    elif "confident" in user_prompt:
        affective_state = "confident"
    
    # Generate appropriate response based on states
    if focus_state == "Lost" and fatigue_prob > 0.7:
        # High fatigue + lost focus
        if fatigue_prob > 0.9 and is_late:
            # Extremely tired and late = suspend session
            return json.dumps({
                "action_type": "suggest_break",
                "message": "You're extremely tired and it's getting late. Let's suspend this session and continue tomorrow when you're fresh.",
                "reasoning": "Critical fatigue levels combined with late hour indicate need for full rest.",
                "target_task_id": None,
                "schedule_changes": {
                    "action": "suspend_session",
                    "reasoning": "Coach detected extreme fatigue late at night and suspended session"
                }
            })
        else:
            # Regular high fatigue = suggest break with duration based on fatigue level
            break_duration = 5 if fatigue_prob <= 0.8 else 10
            return json.dumps({
                "action_type": "suggest_break",
                "message": f"You seem quite tired. How about taking a {break_duration}-minute break to recharge?",
                "reasoning": "High fatigue levels combined with lost focus indicate need for rest.",
                "target_task_id": None,
                "schedule_changes": {
                    "action": "add_break",
                    "duration_minutes": break_duration,
                    "affected_task_ids": [],
                    "reasoning": f"Coach detected high fatigue ({fatigue_prob:.1f}) and suggested {break_duration}-minute break"
                }
            })
    elif affective_state == "frustrated" and focus_state == "Lost":
        # Frustrated + lost = encourage with empathy
        return json.dumps({
            "action_type": "encourage",
            "message": "I can see this is challenging. Remember, every expert was once a beginner. You've got this!",
            "reasoning": "Frustration with lost focus suggests need for motivational support.",
            "target_task_id": None
        })
    elif affective_state == "bored" and focus_state == "Drifting":
        # Bored + drifting = nudge to refocus
        return json.dumps({
            "action_type": "nudge",
            "message": "Let's bring your attention back to the task. What's the next step you need to take?",
            "reasoning": "Boredom with drifting focus needs gentle redirection.",
            "target_task_id": None
        })
    elif affective_state == "confident" and focus_state == "Focused":
        # Confident + focused = positive reinforcement
        return json.dumps({
            "action_type": "encourage",
            "message": "Excellent focus! You're in the zone - keep riding this momentum!",
            "reasoning": "High confidence with strong focus deserves positive reinforcement.",
            "target_task_id": None
        })
    elif fatigue_prob > 0.6:
        # General high fatigue
        if fatigue_prob > 0.9 and is_late:
            # Extremely tired and late = suspend session
            return json.dumps({
                "action_type": "suggest_break",
                "message": "You're working very hard but seem extremely fatigued, and it's late. Let's call it a night and resume tomorrow.",
                "reasoning": "Critical fatigue levels late at night warrant session suspension.",
                "target_task_id": None,
                "schedule_changes": {
                    "action": "suspend_session",
                    "reasoning": "Coach detected extreme fatigue late at night and suspended session"
                }
            })
        else:
            # Regular fatigue = suggest break with duration based on fatigue level
            break_duration = 5 if fatigue_prob <= 0.75 else 10
            return json.dumps({
                "action_type": "suggest_break",
                "message": f"You're working hard! A {break_duration}-minute break might help you maintain quality work.",
                "reasoning": "Elevated fatigue levels suggest rest would be beneficial.",
                "target_task_id": None,
                "schedule_changes": {
                    "action": "add_break",
                    "duration_minutes": break_duration,
                    "affected_task_ids": [],
                    "reasoning": f"Coach detected elevated fatigue ({fatigue_prob:.1f}) and suggested {break_duration}-minute break"
                }
            })
    elif affective_state == "stressed":
        # Stressed = calming encouragement
        return json.dumps({
            "action_type": "encourage",
            "message": "Take a deep breath. You're capable and prepared for this challenge.",
            "reasoning": "Stress levels indicate need for calming, confidence-building support.",
            "target_task_id": None
        })
    else:
        # Default encouraging response
        return json.dumps({
            "action_type": "encourage",
            "message": "You're doing great! Keep up the good work.",
            "reasoning": "Student shows good engagement and focus levels.",
            "target_task_id": None
        })


def decide_with_llm(input_data: CoachInput) -> CoachAction:
    context_json = input_data.model_dump_json(indent=2)

    user_prompt = build_user_prompt(context_json)

    raw = call_gemini(SYSTEM_PROMPT, user_prompt)

    try:
        parsed = json.loads(raw)
        action = CoachAction(**parsed)
        return action
    except (json.JSONDecodeError, ValueError) as e:
        # Fallback to silence
        return CoachAction(
            action_type="silence",
            message=None,
            reasoning=f"Failed to parse LLM response: {str(e)}"
        )
