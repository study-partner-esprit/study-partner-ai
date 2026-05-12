"""
Reward engine for computing rewards on mastery confirmation.
"""

from typing import Dict, List, Any

from src.evaluator.schemas import TaskEvaluationContext


class RewardEngine:
    """Computes rewards for successful evaluations (in-memory, no database)."""

    # Base learning points
    BASE_POINTS = 100

    @staticmethod
    def compute_reward(
        mastery_score: float,
        context: TaskEvaluationContext,
        concepts_covered: List[str],
    ) -> Dict[str, Any]:
        """
        Compute reward for mastery confirmation.

        Args:
            mastery_score: Final mastery score (>= 0.85)
            context: Task evaluation context
            concepts_covered: Concepts student demonstrated

        Returns:
            Reward dict with learning_points, streak_increment, concepts
        """
        # Base points (no difficulty levels in in-memory context)
        base = RewardEngine.BASE_POINTS

        # Bonus for high score (only positive bonus)
        score_bonus = max(0, int((mastery_score - 0.85) * 200))

        # Total learning points
        learning_points = base + score_bonus

        # Streak increment based on score
        if mastery_score >= 0.95:
            streak_increment = 3
        elif mastery_score >= 0.90:
            streak_increment = 2
        else:
            streak_increment = 1

        return {
            "learning_points": learning_points,
            "streak_increment": streak_increment,
            "concepts_covered": concepts_covered[:3],  # Top 3 concepts
        }

    @staticmethod
    def compute_reschedule_recommendation(
        mastery_score: float,
        weak_concepts: List[str],
    ) -> str:
        """
        Recommend action for rescheduling.

        Args:
            mastery_score: Final mastery score (< 0.60)
            weak_concepts: List of weak concepts

        Returns:
            Recommended action: "BREAK_DOWN", "REVIEW", or "SIMPLIFY"
        """
