"""
Trend analysis service for weekly progression tracking.
"""

import logging
from src.reflection.services.aggregation_service import get_all_weeks_summary

logger = logging.getLogger(__name__)


def compute_trends(user_id: str) -> dict:
    """
    Analyze progression trends across weeks.
    Returns trends and an overall progression score.
    """
    weeks = get_all_weeks_summary(user_id)

    if not weeks or (isinstance(weeks[0], dict) and weeks[0].get("status") == "error"):
        return {"status": "error", "detail": "No data to analyze"}

    if len(weeks) < 2:
        return {
            "user_id": user_id,
            "status": "insufficient_data",
            "message": "At least 2 weeks of data needed for trend analysis",
            "weeks_available": len(weeks)
        }

    # Compare latest week to previous week
    latest = weeks[-1]
    previous = weeks[-2]

    def trend_label(current, previous, higher_is_better=True):
        """Determine trend label based on percentage change."""
        if previous == 0:
            return "stable"
        change = (current - previous) / previous * 100
        if higher_is_better:
            if change > 5:
                return "improving"
            if change < -5:
                return "declining"
            return "stable"
        else:
            # For fatigue, lower is better
            if change < -5:
                return "improving"
            if change > 5:
                return "declining"
            return "stable"

    def change_percent(current, previous):
        if previous == 0:
            return 0.0
        return round((current - previous) / previous * 100, 1)

    focus_trend = trend_label(latest["avg_focus_score"], previous["avg_focus_score"], higher_is_better=True)
    fatigue_trend = trend_label(latest["avg_fatigue_score"], previous["avg_fatigue_score"], higher_is_better=False)
    xp_trend = trend_label(latest["total_xp_earned"], previous["total_xp_earned"], higher_is_better=True)
    minutes_trend = trend_label(latest["total_study_minutes"], previous["total_study_minutes"], higher_is_better=True)

    # Global progression score (0-100)
    score_map = {"improving": 2, "stable": 1, "declining": 0}
    raw_score = (
        score_map[focus_trend] +
        score_map[fatigue_trend] +
        score_map[xp_trend] +
        score_map[minutes_trend]
    )
    progression_score = round((raw_score / 8) * 100)

    return {
        "user_id": user_id,
        "weeks_analyzed": len(weeks),
        "latest_week": {"year": latest["year"], "week": latest["week"]},
        "previous_week": {"year": previous["year"], "week": previous["week"]},
        "progression_score": progression_score,
        "trends": {
            "focus": {
                "trend": focus_trend,
                "current": latest["avg_focus_score"],
                "previous": previous["avg_focus_score"],
                "change_percent": change_percent(latest["avg_focus_score"], previous["avg_focus_score"])
            },
            "fatigue": {
                "trend": fatigue_trend,
                "current": latest["avg_fatigue_score"],
                "previous": previous["avg_fatigue_score"],
                "change_percent": change_percent(latest["avg_fatigue_score"], previous["avg_fatigue_score"])
            },
            "xp": {
                "trend": xp_trend,
                "current": latest["total_xp_earned"],
                "previous": previous["total_xp_earned"],
                "change_percent": change_percent(latest["total_xp_earned"], previous["total_xp_earned"])
            },
            "study_minutes": {
                "trend": minutes_trend,
                "current": latest["total_study_minutes"],
                "previous": previous["total_study_minutes"],
                "change_percent": change_percent(latest["total_study_minutes"], previous["total_study_minutes"])
            }
        }
    }