"""Feasibility checking for plans."""

from typing import List


def is_plan_feasible(tasks: List, available_minutes: int) -> bool:
    """Check if a plan is feasible given available time.

    Args:
        tasks: List of tasks to check
        available_minutes: Available time in minutes

    Returns:
        True if plan is feasible, False otherwise
    """
    total_estimated_time = sum(task.estimated_minutes for task in tasks)
    return total_estimated_time <= available_minutes
