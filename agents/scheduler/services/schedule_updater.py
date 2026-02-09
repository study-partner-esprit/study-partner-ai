"""
Schedule Updater Service
Handles dynamic schedule modifications requested by the Coach agent.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pymongo import MongoClient

from agents.coach.models.schemas import ScheduleChange


class ScheduleUpdater:
    """
    Service for applying schedule changes requested by the Coach agent.
    Updates the task_scheduling collection in MongoDB.
    """

    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db_name = os.getenv("DB_NAME", "study_partner")
        self.collection_name = os.getenv("TASK_SCHEDULING_COLLECTION", "task_scheduling")

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

    def apply_schedule_change(self, schedule_change: ScheduleChange) -> bool:
        """
        Apply a schedule change to the database.

        Args:
            schedule_change: The schedule change requested by Coach

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get the latest scheduling document
            doc = self.collection.find_one(sort=[("_id", -1)])
            if not doc:
                print("No scheduling document found")
                return False

            if schedule_change.action == "add_break":
                return self._add_break(doc, schedule_change)
            elif schedule_change.action == "extend_task":
                return self._extend_task(doc, schedule_change)
            elif schedule_change.action == "reschedule_task":
                return self._reschedule_task(doc, schedule_change)
            elif schedule_change.action == "cancel_task":
                return self._cancel_task(doc, schedule_change)
            elif schedule_change.action == "suspend_session":
                return self._suspend_session(doc, schedule_change)
            else:
                print(f"Unknown schedule action: {schedule_change.action}")
                return False

        except Exception as e:
            print(f"Error applying schedule change: {e}")
            return False

    def _add_break(self, doc: Dict[str, Any], change: ScheduleChange) -> bool:
        """Add a break slot to the schedule."""
        if not change.duration_minutes:
            return False

        sessions = doc.get("sessions", [])
        current_time = datetime.now()

        # Create a break session
        break_session = {
            "task_id": f"break_{int(current_time.timestamp())}",
            "start_datetime": current_time.isoformat(),
            "end_datetime": (current_time + timedelta(minutes=change.duration_minutes)).isoformat(),
            "break_after_minutes": 0,
            "slot_score": 1.0,
            "scheduled": True
        }

        # Insert break at the beginning (immediate break)
        sessions.insert(0, break_session)

        # Shift all subsequent sessions by the break duration
        for i in range(1, len(sessions)):
            session = sessions[i]
            # Handle both string and datetime formats
            if isinstance(session["start_datetime"], str):
                start_time = datetime.fromisoformat(session["start_datetime"])
                end_time = datetime.fromisoformat(session["end_datetime"])
            else:
                start_time = session["start_datetime"]
                end_time = session["end_datetime"]

            # Shift this session
            session["start_datetime"] = (start_time + timedelta(minutes=change.duration_minutes)).isoformat()
            session["end_datetime"] = (end_time + timedelta(minutes=change.duration_minutes)).isoformat()

        # Save the updated document
        self.collection.replace_one({"_id": doc["_id"]}, doc)
        print(f"Added {change.duration_minutes}-minute break and shifted subsequent tasks")
        return True

        return False

    def _reschedule_task(self, doc: Dict[str, Any], change: ScheduleChange) -> bool:
        """Reschedule a task to a new time."""
        if not change.new_start_time or not change.affected_task_ids:
            return False

        sessions = doc.get("sessions", [])
        task_id = change.affected_task_ids[0]

        for session in sessions:
            if session["task_id"] == task_id:
                # Calculate duration
                start_time = datetime.fromisoformat(session["start_datetime"])
                end_time = datetime.fromisoformat(session["end_datetime"])
                duration = end_time - start_time

                # Set new times
                session["start_datetime"] = change.new_start_time.isoformat()
                session["end_datetime"] = (change.new_start_time + duration).isoformat()

                self.collection.replace_one({"_id": doc["_id"]}, doc)
                print(f"Rescheduled task {task_id} to {change.new_start_time}")
                return True

        return False

    def _cancel_task(self, doc: Dict[str, Any], change: ScheduleChange) -> bool:
        """Cancel/remove a task from the schedule."""
        if not change.affected_task_ids:
            return False

        sessions = doc.get("sessions", [])
        task_id = change.affected_task_ids[0]

        # Remove the task
        original_length = len(sessions)
        sessions[:] = [s for s in sessions if s["task_id"] != task_id]

        if len(sessions) < original_length:
            self.collection.replace_one({"_id": doc["_id"]}, doc)
            print(f"Cancelled task {task_id}")
            return True

        return False

    def _suspend_session(self, doc: Dict[str, Any], change: ScheduleChange) -> bool:
        """Suspend the current study session until tomorrow."""
        # Add a suspension marker to the document
        doc["suspended"] = True
        doc["suspended_at"] = datetime.now().isoformat()
        doc["resume_date"] = (datetime.now() + timedelta(days=1)).date().isoformat()

        self.collection.replace_one({"_id": doc["_id"]}, doc)
        print(f"Suspended study session until tomorrow ({change.reasoning})")
        return True