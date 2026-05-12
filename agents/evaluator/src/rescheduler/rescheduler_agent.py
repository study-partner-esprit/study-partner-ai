"""
Rescheduler agent for handling failed evaluations.
"""

import logging
from typing import Dict, Any

from src.database.repositories.scheduling_repository import SchedulingRepository
from src.evaluator.schemas import ReschedulePayload

logger = logging.getLogger(__name__)


class ReschedulerAgent:
    """Handles rescheduling logic for failed evaluations."""

    def __init__(self, scheduling_repo: SchedulingRepository):
        self.scheduling_repo = scheduling_repo

    def handle_reschedule(self, task_id: str, reschedule_payload: ReschedulePayload) -> Dict[str, Any]:
        """
        Handle reschedule recommendation.

        Args:
            task_id: Task identifier
            reschedule_payload: Reschedule details

        Returns:
            Reschedule result
        """
        logger.info(f"Handling reschedule for task {task_id}: {reschedule_payload.action}")

        # Update task status in database
        if self.scheduling_repo:
            self.scheduling_repo.update_task_status(
                task_id=task_id,
                status="rescheduled",
                reschedule_action=reschedule_payload.action,
                weak_concepts=reschedule_payload.weak_concepts,
                misconceptions=reschedule_payload.misconceptions,
            )

        return {
            "task_id": task_id,
            "action": reschedule_payload.action,
            "reason": reschedule_payload.reason,
            "next_steps": self._get_next_steps(reschedule_payload.action),
        }

    def _get_next_steps(self, action: str) -> str:
        """Get next steps based on reschedule action."""
        if action == "REVIEW":
            return "Review the basic concepts and try again later."
        elif action == "SIMPLIFY":
            return "Break down the task into simpler components."
        elif action == "BREAK_DOWN":
            return "Focus on individual concepts before combining them."
        else:
            return "Continue with additional practice."
