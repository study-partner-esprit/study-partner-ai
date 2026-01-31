"""Evaluation utility functions for evaluator agent."""
from typing import Dict, Any, List


def calculate_completion_rate(
    completed_tasks: List[Dict[str, Any]],
    total_tasks: int
) -> float:
    """Calculate task completion rate.
    
    Args:
        completed_tasks: List of completed tasks
        total_tasks: Total number of tasks
        
    Returns:
        Completion rate (0-1)
    """
    if total_tasks == 0:
        return 0.0
    return len(completed_tasks) / total_tasks


def calculate_efficiency_score(
    actual_time: int,
    estimated_time: int,
    quality_score: float
) -> float:
    """Calculate efficiency score based on time and quality.
    
    Args:
        actual_time: Actual time spent in minutes
        estimated_time: Estimated time in minutes
        quality_score: Quality of work (0-1)
        
    Returns:
        Efficiency score (0-1)
    """
    if estimated_time == 0:
        return 0.5
    
    time_efficiency = min(1.0, estimated_time / actual_time)
    overall_efficiency = (time_efficiency * 0.6 + quality_score * 0.4)
    
    return max(0.0, min(1.0, overall_efficiency))


def identify_knowledge_gaps(
    quiz_results: List[Dict[str, Any]],
    threshold: float = 0.7
) -> List[str]:
    """Identify knowledge gaps from quiz results.
    
    Args:
        quiz_results: List of quiz results with scores
        threshold: Minimum acceptable score
        
    Returns:
        List of topics needing improvement
    """
    gaps = []
    
    for result in quiz_results:
        topic = result.get("topic", "Unknown")
        score = result.get("score", 0)
        
        if score < threshold:
            gaps.append(topic)
    
    return gaps


def generate_improvement_plan(
    strengths: List[str],
    weaknesses: List[str],
    time_available: int
) -> List[Dict[str, Any]]:
    """Generate personalized improvement plan.
    
    Args:
        strengths: Areas of strength
        weaknesses: Areas needing improvement
        time_available: Available time in minutes
        
    Returns:
        List of improvement actions
    """
    actions = []
    
    # Prioritize weaknesses
    for weakness in weaknesses[:3]:  # Top 3 weaknesses
        actions.append({
            "action": f"Review {weakness}",
            "priority": "high",
            "estimated_duration": min(30, time_available // len(weaknesses))
        })
    
    # Build on strengths
    for strength in strengths[:2]:  # Top 2 strengths
        actions.append({
            "action": f"Advanced practice in {strength}",
            "priority": "medium",
            "estimated_duration": 20
        })
    
    return actions


def calculate_mastery_level(
    attempt_history: List[float],
    time_decay_factor: float = 0.95
) -> float:
    """Calculate mastery level considering attempt history.
    
    Args:
        attempt_history: List of scores from attempts (0-1)
        time_decay_factor: Factor for older attempts
        
    Returns:
        Mastery level (0-1)
    """
    if not attempt_history:
        return 0.0
    
    weighted_scores = []
    for i, score in enumerate(reversed(attempt_history)):
        weight = time_decay_factor ** i
        weighted_scores.append(score * weight)
    
    total_weight = sum(time_decay_factor ** i for i in range(len(attempt_history)))
    
    return sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
