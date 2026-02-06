from datetime import datetime
from typing import List
import logging

from agents.scheduler.models.time_slot import TimeSlot

logger = logging.getLogger(__name__)


def generate_free_slots(
    day_start: datetime,
    day_end: datetime,
    busy_slots: List[TimeSlot],
    min_slot_minutes: int = 25,
) -> List[TimeSlot]:
    """
    Generate free time slots by subtracting busy slots from work window.
    
    Args:
        day_start: Start of work day
        day_end: End of work day
        busy_slots: List of busy time slots
        min_slot_minutes: Minimum duration for a free slot to be considered
        
    Returns:
        List of free TimeSlot objects, sorted by start time
    """
    busy_slots = sorted(busy_slots, key=lambda s: s.start)

    free = []
    cursor = day_start

    for b in busy_slots:
        if b.start > cursor:
            delta = (b.start - cursor).total_seconds() / 60
            if delta >= min_slot_minutes:
                free.append(TimeSlot(start=cursor, end=b.start))

        cursor = max(cursor, b.end)

    if cursor < day_end:
        delta = (day_end - cursor).total_seconds() / 60
        if delta >= min_slot_minutes:
            free.append(TimeSlot(start=cursor, end=day_end))

    logger.debug(
        f"Generated {len(free)} free slots for {day_start.date()} "
        f"(work window {day_start.time()}-{day_end.time()})"
    )
    
    return free
