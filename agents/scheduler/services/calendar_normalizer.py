from datetime import datetime
from typing import List
import logging

from agents.scheduler.models.time_slot import TimeSlot

logger = logging.getLogger(__name__)


def normalize_busy_slots(raw_events: List[dict]) -> List[TimeSlot]:
    """
    Normalize raw calendar events into TimeSlot objects.
    
    Args:
        raw_events: List of calendar events with ISO format timestamps.
                   Example: [{"start": "2026-02-06T09:00:00", "end": "2026-02-06T10:30:00"}]
    
    Returns:
        List of TimeSlot objects
    """
    busy = []

    for ev in raw_events:
        try:
            start = datetime.fromisoformat(ev["start"])
            end = datetime.fromisoformat(ev["end"])
            
            if start >= end:
                logger.warning(f"Invalid event: start {start} >= end {end}")
                continue
            
            busy.append(
                TimeSlot(
                    start=start,
                    end=end,
                    score=0.0,
                )
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing event {ev}: {e}")
            continue

    logger.debug(f"Normalized {len(busy)} calendar events")
    return busy
