"""Schedule Orchestrator service for implementing coach recommendations.

This service bridges the gap between Coach decisions and task scheduling,
implementing autonomous schedule adjustments based on ML signals and coaching.
"""

from typing import Optional
from datetime import datetime, timedelta
from pymongo import MongoClient
import os

from agents.coach.models.schemas import CoachAction, ScheduleChange


class ScheduleOrchestrator:
    """
    Orchestrates schedule modifications based on Coach recommendations.
    
    This service:
    1. Monitors CoachAction outputs
    2. Implements schedule changes (breaks, rescheduling, task adjustments)
    3. Updates task_scheduling collection in MongoDB
    4. Provides feedback loop between detection → coaching → scheduling
    """
    
    def __init__(self):
        """Initialize the schedule orchestrator with MongoDB connection."""
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("MONGO_DB_NAME", "study_partner")
        
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.task_scheduling_collection = self.db["task_scheduling"]
        self.study_plan_collection = self.db["study_plans"]
        self.schedule_history_collection = self.db["schedule_history"]
    
    def process_coach_action(
        self, 
        coach_action: CoachAction,
        user_id: str,
        current_time: Optional[datetime] = None
    ) -> dict:
        """
        Process a CoachAction and implement any schedule changes.
        
        Args:
            coach_action: The action recommended by the Coach
            user_id: The user's unique identifier
            current_time: Current timestamp (defaults to now)
        
        Returns:
            Dictionary with implementation status and details
        """
        if current_time is None:
            current_time = datetime.now()
        
        # If no schedule changes requested, return early
        if coach_action.schedule_changes is None:
            return {
                "status": "no_changes",
                "message": "Coach action does not require schedule modifications"
            }
        
        schedule_change = coach_action.schedule_changes
        
        # Route to appropriate handler based on action type
        if schedule_change.action == "add_break":
            return self._add_break(user_id, schedule_change, current_time)
        elif schedule_change.action == "extend_task":
            return self._extend_task(user_id, schedule_change, current_time)
        elif schedule_change.action == "reschedule_task":
            return self._reschedule_task(user_id, schedule_change, current_time)
        elif schedule_change.action == "cancel_task":
            return self._cancel_task(user_id, schedule_change, current_time)
        elif schedule_change.action == "suspend_session":
            return self._suspend_session(user_id, schedule_change, current_time)
        else:
            return {
                "status": "error",
                "message": f"Unknown schedule action: {schedule_change.action}"
            }
    
    def _add_break(
        self, 
        user_id: str, 
        change: ScheduleChange, 
        current_time: datetime
    ) -> dict:
        """Insert a break into the current schedule."""
        try:
            # Get the current task_scheduling document
            schedule_doc = self.task_scheduling_collection.find_one(
                {"user_id": user_id},
                sort=[("_id", -1)]
            )
            
            if not schedule_doc:
                return {"status": "error", "message": "No active schedule found"}
            
            # Find the current or next session
            sessions = schedule_doc.get("sessions", [])
            current_session_idx = None
            for idx, session in enumerate(sessions):
                if session["start_datetime"] <= current_time <= session["end_datetime"]:
                    current_session_idx = idx
                    break
            
            if current_session_idx is None:
                # No current session, find next one
                for idx, session in enumerate(sessions):
                    if session["start_datetime"] > current_time:
                        current_session_idx = idx
                        break
            
            if current_session_idx is None:
                return {"status": "error", "message": "No sessions to insert break into"}
            
            # Create break session
            break_duration = timedelta(minutes=change.duration_minutes or 15)
            break_start = current_time
            break_end = break_start + break_duration
            
            break_session = {
                "task_id": "break",
                "start_datetime": break_start,
                "end_datetime": break_end,
                "duration_minutes": change.duration_minutes or 15,
                "reason": change.reasoning
            }
            
            # Shift subsequent sessions
            time_shift = break_duration
            for idx in range(current_session_idx, len(sessions)):
                sessions[idx]["start_datetime"] += time_shift
                sessions[idx]["end_datetime"] += time_shift
            
            # Insert break
            sessions.insert(current_session_idx, break_session)
            
            # Update schedule
            self.task_scheduling_collection.update_one(
                {"_id": schedule_doc["_id"]},
                {"$set": {"sessions": sessions, "updated_at": current_time}}
            )
            
            # Log the change
            self._log_schedule_change(user_id, "add_break", change, current_time)
            
            return {
                "status": "success",
                "message": f"Added {change.duration_minutes or 15}-minute break",
                "break_start": break_start,
                "break_end": break_end
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to add break: {str(e)}"}
    
    def _extend_task(
        self, 
        user_id: str, 
        change: ScheduleChange, 
        current_time: datetime
    ) -> dict:
        """Extend the duration of the current task."""
        try:
            schedule_doc = self.task_scheduling_collection.find_one(
                {"user_id": user_id},
                sort=[("_id", -1)]
            )
            
            if not schedule_doc:
                return {"status": "error", "message": "No active schedule found"}
            
            sessions = schedule_doc.get("sessions", [])
            current_session_idx = None
            
            # Find current session
            for idx, session in enumerate(sessions):
                if session["start_datetime"] <= current_time <= session["end_datetime"]:
                    current_session_idx = idx
                    break
            
            if current_session_idx is None:
                return {"status": "error", "message": "No current session to extend"}
            
            # Extend current session
            extension = timedelta(minutes=change.duration_minutes or 15)
            sessions[current_session_idx]["end_datetime"] += extension
            sessions[current_session_idx]["duration_minutes"] += (change.duration_minutes or 15)
            
            # Shift subsequent sessions
            for idx in range(current_session_idx + 1, len(sessions)):
                sessions[idx]["start_datetime"] += extension
                sessions[idx]["end_datetime"] += extension
            
            # Update schedule
            self.task_scheduling_collection.update_one(
                {"_id": schedule_doc["_id"]},
                {"$set": {"sessions": sessions, "updated_at": current_time}}
            )
            
            self._log_schedule_change(user_id, "extend_task", change, current_time)
            
            return {
                "status": "success",
                "message": f"Extended task by {change.duration_minutes or 15} minutes"
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to extend task: {str(e)}"}
    
    def _reschedule_task(
        self, 
        user_id: str, 
        change: ScheduleChange, 
        current_time: datetime
    ) -> dict:
        """Reschedule a specific task to a new time."""
        try:
            schedule_doc = self.task_scheduling_collection.find_one(
                {"user_id": user_id},
                sort=[("_id", -1)]
            )
            
            if not schedule_doc or not change.affected_task_ids:
                return {"status": "error", "message": "Invalid reschedule request"}
            
            sessions = schedule_doc.get("sessions", [])
            task_id = change.affected_task_ids[0]
            
            # Find and remove the task
            task_session = None
            task_idx = None
            for idx, session in enumerate(sessions):
                if session["task_id"] == task_id:
                    task_session = session
                    task_idx = idx
                    break
            
            if not task_session:
                return {"status": "error", "message": f"Task {task_id} not found"}
            
            # Remove from current position
            sessions.pop(task_idx)
            
            # Calculate new position based on new_start_time
            new_start = change.new_start_time or (current_time + timedelta(hours=1))
            task_duration = task_session["end_datetime"] - task_session["start_datetime"]
            
            task_session["start_datetime"] = new_start
            task_session["end_datetime"] = new_start + task_duration
            
            # Find insertion point
            insert_idx = len(sessions)
            for idx, session in enumerate(sessions):
                if session["start_datetime"] > new_start:
                    insert_idx = idx
                    break
            
            sessions.insert(insert_idx, task_session)
            
            # Update schedule
            self.task_scheduling_collection.update_one(
                {"_id": schedule_doc["_id"]},
                {"$set": {"sessions": sessions, "updated_at": current_time}}
            )
            
            self._log_schedule_change(user_id, "reschedule_task", change, current_time)
            
            return {
                "status": "success",
                "message": f"Rescheduled task {task_id} to {new_start}"
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to reschedule: {str(e)}"}
    
    def _cancel_task(
        self, 
        user_id: str, 
        change: ScheduleChange, 
        current_time: datetime
    ) -> dict:
        """Cancel a specific task from the schedule."""
        try:
            schedule_doc = self.task_scheduling_collection.find_one(
                {"user_id": user_id},
                sort=[("_id", -1)]
            )
            
            if not schedule_doc or not change.affected_task_ids:
                return {"status": "error", "message": "Invalid cancel request"}
            
            sessions = schedule_doc.get("sessions", [])
            task_id = change.affected_task_ids[0]
            
            # Remove the task
            sessions = [s for s in sessions if s["task_id"] != task_id]
            
            # Update schedule
            self.task_scheduling_collection.update_one(
                {"_id": schedule_doc["_id"]},
                {"$set": {"sessions": sessions, "updated_at": current_time}}
            )
            
            self._log_schedule_change(user_id, "cancel_task", change, current_time)
            
            return {
                "status": "success",
                "message": f"Cancelled task {task_id}"
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to cancel task: {str(e)}"}
    
    def _suspend_session(
        self, 
        user_id: str, 
        change: ScheduleChange, 
        current_time: datetime
    ) -> dict:
        """Suspend the current study session."""
        try:
            schedule_doc = self.task_scheduling_collection.find_one(
                {"user_id": user_id},
                sort=[("_id", -1)]
            )
            
            if not schedule_doc:
                return {"status": "error", "message": "No active schedule found"}
            
            # Mark schedule as suspended
            self.task_scheduling_collection.update_one(
                {"_id": schedule_doc["_id"]},
                {
                    "$set": {
                        "status": "suspended",
                        "suspended_at": current_time,
                        "suspension_reason": change.reasoning
                    }
                }
            )
            
            self._log_schedule_change(user_id, "suspend_session", change, current_time)
            
            return {
                "status": "success",
                "message": "Study session suspended"
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to suspend: {str(e)}"}
    
    def _log_schedule_change(
        self,
        user_id: str,
        action: str,
        change: ScheduleChange,
        timestamp: datetime
    ):
        """Log schedule changes for audit and analysis."""
        log_entry = {
            "user_id": user_id,
            "action": action,
            "reasoning": change.reasoning,
            "duration_minutes": change.duration_minutes,
            "affected_task_ids": change.affected_task_ids,
            "timestamp": timestamp
        }
        self.schedule_history_collection.insert_one(log_entry)
