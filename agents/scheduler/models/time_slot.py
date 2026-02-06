from datetime import datetime
from pydantic import BaseModel


class TimeSlot(BaseModel):
    start: datetime
    end: datetime
    score: float = 0.0