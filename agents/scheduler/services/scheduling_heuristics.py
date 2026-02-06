import logging

from agents.scheduler.models.time_slot import TimeSlot

logger = logging.getLogger(__name__)


def score_slot(
    slot: TimeSlot,
    historical_productivity: list,
    allow_late_night: bool,
) -> float:
    """
    Score a time slot for scheduling desirability.
    
    Factors:
    - Time of day (morning slots preferred, late night penalized)
    - Historical productivity patterns (if available)
    
    Args:
        slot: TimeSlot to score
        historical_productivity: Historical productivity data (optional)
        allow_late_night: Whether late night scheduling is allowed
        
    Returns:
        Score from 0.0 to 1.0 (higher is better)
    """
    hour = slot.start.hour

    score = 1.0

    # Penalize late night and early morning
    if hour >= 22 or hour < 7:
        if not allow_late_night:
            score -= 0.7
        else:
            score -= 0.3
    # Prefer morning/early afternoon (10-15)
    elif 10 <= hour <= 15:
        score += 0.2

    # Simple productivity heuristic (placeholder for future enhancements)
    if historical_productivity:
        score += 0.1

    return max(0.0, min(1.0, score))  # Clamp to [0.0, 1.0]
