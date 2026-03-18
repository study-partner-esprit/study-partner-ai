from typing import Dict, List, Optional

from agents.scheduler.models.time_slot import TimeSlot
from utils.logger import get_logger

logger = get_logger(__name__)

# Maximum bonus that real focus data can add / subtract relative to the
# neutral baseline (avg focus == 0.5 → +0.0, avg focus == 1.0 → +0.15)
_MAX_FOCUS_BONUS = 0.15


def score_slot(
    slot: TimeSlot,
    historical_productivity: List = [],
    allow_late_night: bool = False,
    hourly_focus_profile: Optional[Dict[int, float]] = None,
) -> float:
    """
    Score a time slot for scheduling desirability.

    Factors:
      1. Time of day  — morning slots preferred, late-night penalised.
      2. Hourly focus profile — real average focus score for this hour-of-day
         derived from *days_back* of historical signal data.  Replaces the
         old placeholder `+0.1` bonus.

    Args:
        slot:                   TimeSlot to score.
        historical_productivity: Legacy param kept for backwards compat (ignored).
        allow_late_night:       Whether late night scheduling is allowed.
        hourly_focus_profile:   Dict[hour_of_day, avg_focus_score] built by
                                SignalRepository.get_hourly_focus_profile().
                                When None (no data), falls back to time-of-day
                                heuristics only.

    Returns:
        Score in [0.0, 1.0] (higher is better).
    """
    hour = slot.start.hour
    score = 1.0

    # --- Time-of-day heuristic ------------------------------------------- #
    if hour >= 22 or hour < 7:
        if not allow_late_night:
            score -= 0.7
        else:
            score -= 0.3
    elif 10 <= hour <= 15:
        score += 0.2

    # --- Hourly focus profile bonus -------------------------------------- #
    if hourly_focus_profile is not None and hour in hourly_focus_profile:
        avg_focus = hourly_focus_profile[hour]
        # Linear map: avg_focus 0.5 → 0.0 bonus, 1.0 → +MAX, 0.0 → -MAX
        focus_bonus = _MAX_FOCUS_BONUS * (avg_focus - 0.5) * 2.0
        score += focus_bonus
        logger.debug(
            "score_slot_focus_bonus",
            extra={
                "hour": hour,
                "avg_focus": round(avg_focus, 3),
                "bonus": round(focus_bonus, 3),
            },
        )

    return max(0.0, min(1.0, score))
