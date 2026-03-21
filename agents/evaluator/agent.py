from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SessionEvaluation:
    score: int
    level: str
    strengths: List[str]
    risks: List[str]
    recommendations: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "level": self.level,
            "strengths": self.strengths,
            "risks": self.risks,
            "recommendations": self.recommendations,
        }


class EvaluatorAgent:
    """Rule-based evaluator for end-of-session quality feedback."""

    def evaluate(
        self,
        *,
        session_duration_minutes: int,
        focus_score: float,
        completed_tasks: int,
        skipped_tasks: int,
    ) -> SessionEvaluation:
        score = 50
        strengths: List[str] = []
        risks: List[str] = []
        recommendations: List[str] = []

        if session_duration_minutes >= 45:
            score += 15
            strengths.append("Maintained a deep-work session length.")
        elif session_duration_minutes >= 25:
            score += 8
            strengths.append("Completed a meaningful study block.")
        else:
            risks.append("Session was short and may not be enough for retention.")
            recommendations.append("Aim for at least 25 minutes in your next session.")

        if focus_score >= 80:
            score += 20
            strengths.append("Excellent sustained focus.")
        elif focus_score >= 60:
            score += 10
            strengths.append("Good focus consistency.")
        else:
            score -= 8
            risks.append("Focus variability was high.")
            recommendations.append("Insert a 5-minute reset break every 25-30 minutes.")

        task_delta = completed_tasks - skipped_tasks
        if task_delta >= 2:
            score += 12
            strengths.append("Strong completion momentum on planned tasks.")
        elif task_delta >= 0:
            score += 5
        else:
            score -= 10
            risks.append("Skipped more tasks than completed.")
            recommendations.append(
                "Reduce task size and prioritize one high-impact task first."
            )

        score = max(0, min(100, int(score)))

        if score >= 80:
            level = "excellent"
        elif score >= 60:
            level = "good"
        elif score >= 40:
            level = "fair"
        else:
            level = "needs_attention"

        if not recommendations:
            recommendations.append("Keep the same routine and increase difficulty gradually.")

        return SessionEvaluation(
            score=score,
            level=level,
            strengths=strengths,
            risks=risks,
            recommendations=recommendations,
        )
