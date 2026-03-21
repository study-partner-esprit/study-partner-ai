"""Schedule Orchestrator service for implementing coach recommendations.

This service bridges the gap between Coach decisions and task scheduling,
implementing autonomous schedule adjustments based on ML signals and coaching.
Uses MongoDB transactions to ensure atomic writes across collections.
"""

from typing import Optional
from datetime import datetime, timedelta
from pymongo import MongoClient
import os

from agents.coach.models.schemas import CoachAction, ScheduleChange
from utils.logger import get_logger

logger = get_logger(__name__)


class ScheduleOrchestrator:
    """
    Orchestrates schedule modifications based on Coach recommendations.

    This service:
    1. Monitors CoachAction outputs
    2. Implements schedule changes (breaks, rescheduling, task adjustments)
    3. Updates task_scheduling collection in MongoDB
    4. Provides feedback loop between detection → coaching → scheduling

    All multi-collection writes use MongoDB sessions/transactions for atomicity.
    """

    def __init__(self):
        """Initialize the schedule orchestrator with MongoDB connection."""
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("MONGO_DB_NAME", "study_partner")

        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.task_scheduling_collection = self.db["task_scheduling"]
        self.study_plan_collection = self.db["studyplans"]
        self.schedule_history_collection = self.db["schedule_history"]
        self.schedule_snapshots_collection = self.db["schedule_snapshots"]

    def process_coach_action(
        self,
        coach_action: CoachAction,
        user_id: str,
        current_time: Optional[datetime] = None,
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
                "message": "Coach action does not require schedule modifications",
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
                "message": f"Unknown schedule action: {schedule_change.action}",
            }

    def _add_break(
        self, user_id: str, change: ScheduleChange, current_time: datetime
    ) -> dict:
        """Insert a break into the current schedule (transactional)."""
        with self.client.start_session() as session:
            try:
                with session.start_transaction():
                    schedule_doc = self.task_scheduling_collection.find_one(
                        {"user_id": user_id},
                        sort=[("_id", -1)],
                        session=session,
                    )

                    if not schedule_doc:
                        return {
                            "status": "error",
                            "message": "No active schedule found",
                        }

                    sessions_list = schedule_doc.get("sessions", [])
                    current_session_idx = None
                    for idx, s in enumerate(sessions_list):
                        if s["start_datetime"] <= current_time <= s["end_datetime"]:
                            current_session_idx = idx
                            break

                    if current_session_idx is None:
                        for idx, s in enumerate(sessions_list):
                            if s["start_datetime"] > current_time:
                                current_session_idx = idx
                                break

                    if current_session_idx is None:
                        return {
                            "status": "error",
                            "message": "No sessions to insert break into",
                        }

                    break_duration = timedelta(minutes=change.duration_minutes or 15)
                    break_start = current_time
                    break_end = break_start + break_duration

                    break_session = {
                        "task_id": "break",
                        "start_datetime": break_start,
                        "end_datetime": break_end,
                        "duration_minutes": change.duration_minutes or 15,
                        "reason": change.reasoning,
                    }

                    time_shift = break_duration
                    for idx in range(current_session_idx, len(sessions_list)):
                        sessions_list[idx]["start_datetime"] += time_shift
                        sessions_list[idx]["end_datetime"] += time_shift

                    sessions_list.insert(current_session_idx, break_session)

                    self.task_scheduling_collection.update_one(
                        {"_id": schedule_doc["_id"]},
                        {
                            "$set": {
                                "sessions": sessions_list,
                                "updated_at": current_time,
                            }
                        },
                        session=session,
                    )

                    self._log_schedule_change(
                        user_id, "add_break", change, current_time, session=session
                    )

                    self._persist_schedule_snapshot(
                        user_id,
                        schedule_doc.get("_id"),
                        sessions_list,
                        "add_break",
                        current_time,
                        session=session,
                    )

                return {
                    "status": "success",
                    "message": f"Added {change.duration_minutes or 15}-minute break",
                    "break_start": break_start,
                    "break_end": break_end,
                }
            except Exception as e:
                logger.error(
                    "schedule_add_break_error",
                    extra={"error": str(e), "user_id": user_id},
                )
                return {"status": "error", "message": f"Failed to add break: {str(e)}"}

    def _extend_task(
        self, user_id: str, change: ScheduleChange, current_time: datetime
    ) -> dict:
        """Extend the duration of the current task (transactional)."""
        with self.client.start_session() as session:
            try:
                with session.start_transaction():
                    schedule_doc = self.task_scheduling_collection.find_one(
                        {"user_id": user_id},
                        sort=[("_id", -1)],
                        session=session,
                    )

                    if not schedule_doc:
                        return {
                            "status": "error",
                            "message": "No active schedule found",
                        }

                    sessions_list = schedule_doc.get("sessions", [])
                    current_session_idx = None

                    for idx, s in enumerate(sessions_list):
                        if s["start_datetime"] <= current_time <= s["end_datetime"]:
                            current_session_idx = idx
                            break

                    if current_session_idx is None:
                        return {
                            "status": "error",
                            "message": "No current session to extend",
                        }

                    extension = timedelta(minutes=change.duration_minutes or 15)
                    sessions_list[current_session_idx]["end_datetime"] += extension
                    sessions_list[current_session_idx]["duration_minutes"] += (
                        change.duration_minutes or 15
                    )

                    for idx in range(current_session_idx + 1, len(sessions_list)):
                        sessions_list[idx]["start_datetime"] += extension
                        sessions_list[idx]["end_datetime"] += extension

                    self.task_scheduling_collection.update_one(
                        {"_id": schedule_doc["_id"]},
                        {
                            "$set": {
                                "sessions": sessions_list,
                                "updated_at": current_time,
                            }
                        },
                        session=session,
                    )

                    self._log_schedule_change(
                        user_id, "extend_task", change, current_time, session=session
                    )

                    self._persist_schedule_snapshot(
                        user_id,
                        schedule_doc.get("_id"),
                        sessions_list,
                        "extend_task",
                        current_time,
                        session=session,
                    )

                return {
                    "status": "success",
                    "message": f"Extended task by {change.duration_minutes or 15} minutes",
                }
            except Exception as e:
                logger.error(
                    "schedule_extend_task_error",
                    extra={"error": str(e), "user_id": user_id},
                )
                return {
                    "status": "error",
                    "message": f"Failed to extend task: {str(e)}",
                }

    def _reschedule_task(
        self, user_id: str, change: ScheduleChange, current_time: datetime
    ) -> dict:
        """Reschedule a specific task to a new time (transactional)."""
        with self.client.start_session() as session:
            try:
                with session.start_transaction():
                    schedule_doc = self.task_scheduling_collection.find_one(
                        {"user_id": user_id},
                        sort=[("_id", -1)],
                        session=session,
                    )

                    if not schedule_doc or not change.affected_task_ids:
                        return {
                            "status": "error",
                            "message": "Invalid reschedule request",
                        }

                    sessions_list = schedule_doc.get("sessions", [])
                    task_id = change.affected_task_ids[0]

                    task_session = None
                    task_idx = None
                    for idx, s in enumerate(sessions_list):
                        if s["task_id"] == task_id:
                            task_session = s
                            task_idx = idx
                            break

                    if not task_session:
                        return {
                            "status": "error",
                            "message": f"Task {task_id} not found",
                        }

                    sessions_list.pop(task_idx)

                    new_start = change.new_start_time or (
                        current_time + timedelta(hours=1)
                    )
                    task_duration = (
                        task_session["end_datetime"] - task_session["start_datetime"]
                    )

                    task_session["start_datetime"] = new_start
                    task_session["end_datetime"] = new_start + task_duration

                    insert_idx = len(sessions_list)
                    for idx, s in enumerate(sessions_list):
                        if s["start_datetime"] > new_start:
                            insert_idx = idx
                            break

                    sessions_list.insert(insert_idx, task_session)

                    self.task_scheduling_collection.update_one(
                        {"_id": schedule_doc["_id"]},
                        {
                            "$set": {
                                "sessions": sessions_list,
                                "updated_at": current_time,
                            }
                        },
                        session=session,
                    )

                    self._log_schedule_change(
                        user_id,
                        "reschedule_task",
                        change,
                        current_time,
                        session=session,
                    )

                    self._persist_schedule_snapshot(
                        user_id,
                        schedule_doc.get("_id"),
                        sessions_list,
                        "reschedule_task",
                        current_time,
                        session=session,
                    )

                return {
                    "status": "success",
                    "message": f"Rescheduled task {task_id} to {new_start}",
                }
            except Exception as e:
                logger.error(
                    "schedule_reschedule_error",
                    extra={"error": str(e), "user_id": user_id},
                )
                return {"status": "error", "message": f"Failed to reschedule: {str(e)}"}

    def _cancel_task(
        self, user_id: str, change: ScheduleChange, current_time: datetime
    ) -> dict:
        """Cancel a specific task from the schedule (transactional)."""
        with self.client.start_session() as session:
            try:
                with session.start_transaction():
                    schedule_doc = self.task_scheduling_collection.find_one(
                        {"user_id": user_id},
                        sort=[("_id", -1)],
                        session=session,
                    )

                    if not schedule_doc or not change.affected_task_ids:
                        return {"status": "error", "message": "Invalid cancel request"}

                    sessions_list = schedule_doc.get("sessions", [])
                    task_id = change.affected_task_ids[0]

                    sessions_list = [
                        s for s in sessions_list if s["task_id"] != task_id
                    ]

                    self.task_scheduling_collection.update_one(
                        {"_id": schedule_doc["_id"]},
                        {
                            "$set": {
                                "sessions": sessions_list,
                                "updated_at": current_time,
                            }
                        },
                        session=session,
                    )

                    self._log_schedule_change(
                        user_id, "cancel_task", change, current_time, session=session
                    )

                    self._persist_schedule_snapshot(
                        user_id,
                        schedule_doc.get("_id"),
                        sessions_list,
                        "cancel_task",
                        current_time,
                        session=session,
                    )

                return {"status": "success", "message": f"Cancelled task {task_id}"}
            except Exception as e:
                logger.error(
                    "schedule_cancel_task_error",
                    extra={"error": str(e), "user_id": user_id},
                )
                return {
                    "status": "error",
                    "message": f"Failed to cancel task: {str(e)}",
                }

    def _suspend_session(
        self, user_id: str, change: ScheduleChange, current_time: datetime
    ) -> dict:
        """Suspend the current study session (transactional)."""
        with self.client.start_session() as session:
            try:
                with session.start_transaction():
                    schedule_doc = self.task_scheduling_collection.find_one(
                        {"user_id": user_id},
                        sort=[("_id", -1)],
                        session=session,
                    )

                    if not schedule_doc:
                        return {
                            "status": "error",
                            "message": "No active schedule found",
                        }

                    self.task_scheduling_collection.update_one(
                        {"_id": schedule_doc["_id"]},
                        {
                            "$set": {
                                "status": "suspended",
                                "suspended_at": current_time,
                                "suspension_reason": change.reasoning,
                            }
                        },
                        session=session,
                    )

                    self._log_schedule_change(
                        user_id,
                        "suspend_session",
                        change,
                        current_time,
                        session=session,
                    )

                return {"status": "success", "message": "Study session suspended"}
            except Exception as e:
                logger.error(
                    "schedule_suspend_error",
                    extra={"error": str(e), "user_id": user_id},
                )
                return {"status": "error", "message": f"Failed to suspend: {str(e)}"}

    def get_schedule_status(self, user_id: str) -> dict:
        """Return current schedule status and latest snapshot metadata."""
        schedule_doc = self.task_scheduling_collection.find_one(
            {"user_id": user_id}, sort=[("_id", -1)]
        )
        if not schedule_doc:
            return {
                "status": "missing",
                "message": "No schedule found for user",
                "user_id": user_id,
                "sessions": [],
            }

        sessions = schedule_doc.get("sessions", [])
        now = datetime.now()
        upcoming_count = sum(
            1
            for s in sessions
            if s.get("start_datetime") and s["start_datetime"] > now
        )
        completed_count = sum(
            1 for s in sessions if s.get("end_datetime") and s["end_datetime"] < now
        )

        latest_snapshot = self.schedule_snapshots_collection.find_one(
            {"user_id": user_id}, sort=[("created_at", -1)]
        )

        return {
            "status": "ok",
            "user_id": user_id,
            "schedule_id": str(schedule_doc.get("_id")),
            "schedule_state": schedule_doc.get("status", "active"),
            "total_sessions": len(sessions),
            "upcoming_sessions": upcoming_count,
            "completed_sessions": completed_count,
            "updated_at": schedule_doc.get("updated_at"),
            "latest_snapshot": {
                "reason": latest_snapshot.get("reason") if latest_snapshot else None,
                "created_at": latest_snapshot.get("created_at") if latest_snapshot else None,
            },
            "sessions": sessions,
        }

    def optimize_schedule(self, user_id: str, reason: str = "manual_optimize") -> dict:
        """Perform a lightweight schedule optimization by ordering sessions by start time."""
        with self.client.start_session() as session:
            with session.start_transaction():
                schedule_doc = self.task_scheduling_collection.find_one(
                    {"user_id": user_id}, sort=[("_id", -1)], session=session
                )

                if not schedule_doc:
                    return {
                        "status": "missing",
                        "message": "No schedule found for user",
                        "user_id": user_id,
                    }

                sessions = schedule_doc.get("sessions", [])
                sorted_sessions = sorted(
                    sessions,
                    key=lambda s: s.get("start_datetime") or datetime.max,
                )

                self.task_scheduling_collection.update_one(
                    {"_id": schedule_doc["_id"]},
                    {
                        "$set": {
                            "sessions": sorted_sessions,
                            "updated_at": datetime.now(),
                        }
                    },
                    session=session,
                )

                self._persist_schedule_snapshot(
                    user_id,
                    schedule_doc.get("_id"),
                    sorted_sessions,
                    reason,
                    datetime.now(),
                    session=session,
                )

                return {
                    "status": "ok",
                    "message": "Schedule optimized",
                    "user_id": user_id,
                    "total_sessions": len(sorted_sessions),
                }

    def _log_schedule_change(
        self,
        user_id: str,
        action: str,
        change: ScheduleChange,
        timestamp: datetime,
        session=None,
    ):
        """Log schedule changes for audit and analysis."""
        log_entry = {
            "user_id": user_id,
            "action": action,
            "reasoning": change.reasoning,
            "duration_minutes": change.duration_minutes,
            "affected_task_ids": change.affected_task_ids,
            "timestamp": timestamp,
        }
        self.schedule_history_collection.insert_one(log_entry, session=session)

    def _persist_schedule_snapshot(
        self,
        user_id: str,
        schedule_id,
        sessions: list,
        reason: str,
        timestamp: datetime,
        session=None,
    ):
        """Persist a snapshot of schedule state for recovery and auditing."""
        doc = {
            "user_id": user_id,
            "schedule_id": schedule_id,
            "reason": reason,
            "sessions": sessions,
            "created_at": timestamp,
        }
        self.schedule_snapshots_collection.insert_one(doc, session=session)
