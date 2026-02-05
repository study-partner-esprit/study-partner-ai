"""Clarification checker stub."""


class ClarificationChecker:
    """Stub clarification checker."""

    def check_goal(self, goal):
        """Stub check_goal method - returns True if goal is too vague."""
        return len(goal.split()) < 3  # Consider goals with < 3 words as vague
