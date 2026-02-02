from typing import List
from agents.planner.models.task_graph import AtomicTask

class ClarificationChecker:
    """
    Checks if a plan needs clarification or deadline negotiation
    """

    MIN_GOAL_LENGTH = 10  # minimum characters for a goal to be considered specific

    def check_goal(self, goal: str) -> bool:
        """
        Return True if goal is too vague and requires clarification
        """
        if len(goal.strip()) < self.MIN_GOAL_LENGTH:
            return True
        return False

    def check_plan_feasibility(self, tasks: List[AtomicTask], available_minutes: int) -> bool:
        """
        Return True if the plan exceeds available time and needs negotiation
        """
        total_minutes = sum(t.estimated_minutes for t in tasks)
        return total_minutes > available_minutes
