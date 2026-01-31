"""Task utility functions for planner agent."""
from typing import List, Dict, Any


def estimate_task_duration(
    task_description: str,
    difficulty: str,
    user_level: str
) -> int:
    """Estimate task duration based on various factors.
    
    Args:
        task_description: Description of the task
        difficulty: Task difficulty level
        user_level: User's proficiency level
        
    Returns:
        Estimated duration in minutes
    """
    # TODO: Implement actual estimation logic
    base_duration = 30
    
    difficulty_multiplier = {
        "easy": 0.7,
        "medium": 1.0,
        "hard": 1.5
    }
    
    return int(base_duration * difficulty_multiplier.get(difficulty, 1.0))


def prioritize_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prioritize tasks based on dependencies and importance.
    
    Args:
        tasks: List of tasks to prioritize
        
    Returns:
        Sorted list of tasks
    """
    # TODO: Implement sophisticated prioritization algorithm
    return sorted(tasks, key=lambda t: t.get("priority", "medium"))


def split_large_task(
    task: Dict[str, Any],
    max_duration: int = 60
) -> List[Dict[str, Any]]:
    """Split a large task into smaller subtasks.
    
    Args:
        task: Task to split
        max_duration: Maximum duration for a subtask in minutes
        
    Returns:
        List of subtasks
    """
    # TODO: Implement task splitting logic
    duration = task.get("estimated_duration", 30)
    
    if duration <= max_duration:
        return [task]
    
    # Simple split for now
    num_parts = (duration + max_duration - 1) // max_duration
    subtasks = []
    
    for i in range(num_parts):
        subtask = task.copy()
        subtask["title"] = f"{task['title']} - Part {i+1}"
        subtask["estimated_duration"] = min(max_duration, duration - i * max_duration)
        subtasks.append(subtask)
    
    return subtasks
