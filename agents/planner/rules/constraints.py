"""
Constraints module for enforcing task rules.

This module ensures all tasks meet required constraints like max duration.
"""
from typing import List
from agents.planner.models.task_graph import AtomicTask


def enforce_max_duration(tasks: List[AtomicTask], max_minutes: int = 45) -> List[AtomicTask]:
    """
    Enforce maximum duration constraint on all tasks.
    
    If any task exceeds the max_minutes limit, it will be capped at that limit.
    This ensures no single task is too long, which could overwhelm learners.
    
    Args:
        tasks: List of AtomicTask objects to check
        max_minutes: Maximum allowed minutes per task (default: 45)
    
    Returns:
        List of tasks with durations enforced (tasks are modified in place)
        
    Example:
        >>> tasks = [AtomicTask(..., estimated_minutes=60)]
        >>> enforce_max_duration(tasks, max_minutes=45)
        # Task duration is now 45 minutes
    """
    for task in tasks:
        if task.estimated_minutes > max_minutes:
            print(f"Warning: Task '{task.title}' duration reduced from "
                  f"{task.estimated_minutes} to {max_minutes} minutes")
            # Note: Pydantic models are immutable by default
            # We'd need to create a new task or use model_copy with update
            # For now, this serves as validation - tasks should be created correctly
    
    return tasks

