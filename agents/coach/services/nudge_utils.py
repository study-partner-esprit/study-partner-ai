"""Nudge utility functions for coach agent."""
from typing import Dict, Any, List


def generate_nudge(
    user_state: Dict[str, Any],
    task_difficulty: str,
    time_on_task: int
) -> str:
    """Generate contextual nudge message.
    
    Args:
        user_state: Current user state (focus, motivation, etc.)
        task_difficulty: Current task difficulty
        time_on_task: Time spent on current task in minutes
        
    Returns:
        Nudge message string
    """
    # TODO: Implement intelligent nudge generation
    motivation_level = user_state.get("motivation", 0.5)
    
    if motivation_level < 0.3:
        return "You've got this! Take a deep breath and focus on one step at a time."
    elif time_on_task > 45:
        return "Great focus! Consider taking a short break to refresh your mind."
    else:
        return "You're making excellent progress! Keep up the great work."


def suggest_break(
    session_duration: int,
    last_break: int,
    fatigue_score: float
) -> Dict[str, Any]:
    """Suggest break timing based on various factors.
    
    Args:
        session_duration: Total session duration in minutes
        last_break: Minutes since last break
        fatigue_score: Fatigue score (0-1)
        
    Returns:
        Break suggestion with timing and duration
    """
    # TODO: Implement smart break suggestion algorithm
    should_break = last_break > 50 or fatigue_score > 0.7
    
    return {
        "should_break": should_break,
        "suggested_duration": 10 if should_break else 0,
        "reason": "fatigue_detected" if fatigue_score > 0.7 else "time_based"
    }


def adjust_difficulty(
    current_difficulty: str,
    success_rate: float,
    user_feedback: str
) -> str:
    """Suggest difficulty adjustment based on performance.
    
    Args:
        current_difficulty: Current task difficulty
        success_rate: User's success rate (0-1)
        user_feedback: User's feedback on difficulty
        
    Returns:
        Suggested difficulty level
    """
    # TODO: Implement adaptive difficulty adjustment
    if success_rate < 0.5 or user_feedback == "too_hard":
        return "easier"
    elif success_rate > 0.9 and user_feedback != "too_easy":
        return "harder"
    else:
        return "maintain"


def generate_encouragement(
    progress: float,
    streak_days: int,
    achievements: List[str]
) -> str:
    """Generate encouraging message based on progress.
    
    Args:
        progress: Progress percentage (0-1)
        streak_days: Number of consecutive study days
        achievements: List of recent achievements
        
    Returns:
        Encouragement message
    """
    if streak_days >= 7:
        return f"Amazing! You've maintained a {streak_days}-day streak! 🔥"
    elif progress >= 0.75:
        return "You're almost there! The finish line is in sight!"
    elif len(achievements) > 0:
        return f"Great job completing '{achievements[-1]}'! Keep going!"
    else:
        return "Every step forward is progress. You're doing great!"
