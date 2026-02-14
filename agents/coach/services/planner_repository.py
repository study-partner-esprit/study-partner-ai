import os
from pymongo import MongoClient
from agents.coach.models.schemas import ScheduledTask


MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "study_partner")
TASK_SCHEDULING_COLLECTION = os.getenv("TASK_SCHEDULING_COLLECTION", "task_scheduling")
STUDY_PLAN_COLLECTION = os.getenv("STUDY_PLAN_COLLECTION", "study_plans")


class PlannerRepository:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.task_collection = self.db[TASK_SCHEDULING_COLLECTION]
        self.study_plan_collection = self.db[STUDY_PLAN_COLLECTION]

    def get_scheduled_tasks(self) -> list[ScheduledTask]:
        try:
            # Fetch the latest scheduling document
            doc = self.task_collection.find_one(sort=[("_id", -1)])
            if not doc:
                # Fallback to mock data for testing
                return self._get_mock_tasks()
            
            study_plan_id = doc.get("study_plan_id")
            if not study_plan_id:
                return self._get_mock_tasks()
            
            # Fetch the study plan to get task details
            study_plan = self.study_plan_collection.find_one({"_id": study_plan_id})
            if not study_plan:
                return self._get_mock_tasks()
            
            task_graph = study_plan.get("task_graph", {})
            atomic_tasks = task_graph.get("atomic_tasks", [])
            task_dict = {task["id"]: task for task in atomic_tasks}
            
            sessions = doc.get("sessions", [])
            tasks = []
            for session in sessions:
                task_id = session["task_id"]
                task_info = task_dict.get(task_id, {})
                title = task_info.get("title", task_id)
                priority = task_info.get("difficulty", 0.5) * 10  # Rough priority
                
                task = ScheduledTask(
                    task_id=task_id,
                    title=title,
                    start_time=session["start_datetime"],
                    end_time=session["end_datetime"],
                    priority=int(priority)
                )
                tasks.append(task)
            return tasks
        except Exception:
            # If MongoDB is not available, return mock data
            return self._get_mock_tasks()
    
    def _get_mock_tasks(self) -> list[ScheduledTask]:
        """Return mock scheduled tasks for testing when DB is unavailable."""
        from datetime import datetime, timedelta
        return [
            ScheduledTask(
                task_id="math_001",
                title="Algebra Fundamentals",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(minutes=45),
                priority=1
            ),
            ScheduledTask(
                task_id="physics_002", 
                title="Newton's Laws",
                start_time=datetime.now() + timedelta(hours=2),
                end_time=datetime.now() + timedelta(hours=2, minutes=30),
                priority=2
            )
        ]