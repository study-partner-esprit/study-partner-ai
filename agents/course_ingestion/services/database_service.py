import os
import traceback
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId

# ----------------------------
# MongoDB setup
# ----------------------------
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "study_partner")
# Use consistent collection names across all services
COLLECTION_NAME = "courses"  # AI-processed courses
STUDY_PLAN_COLLECTION = "studyplans"  # Match Mongoose pluralization
TASK_SCHEDULING_COLLECTION = "task_scheduling"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
study_plan_collection = db[STUDY_PLAN_COLLECTION]


class DatabaseService:
    """Database service for course and study plan operations."""

    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.study_plan_collection = self.db[STUDY_PLAN_COLLECTION]
        self.task_scheduling_collection = self.db[TASK_SCHEDULING_COLLECTION]

    def save_course(self, course: dict):
        """
        Save course to MongoDB.

        Args:
            course (dict): Course JSON following CourseKnowledgeJSON schema
                           (already enriched by the agent)

        Returns:
            str: The inserted document ID
        """
        # Course is already enriched by the ingestion agent
        # Just save it directly to MongoDB
        result = self.collection.insert_one(course)
        return str(result.inserted_id)

    def get_course_by_id(self, course_id: str) -> dict:
        """
        Get a course by ID and return the raw MongoDB document.
        
        Args:
            course_id: The course ID as a string
            
        Returns:
            The raw MongoDB document as a dict
        """
        if isinstance(course_id, str):
            course_id = ObjectId(course_id)
        return self.collection.find_one({"_id": course_id})

    def save_study_plan(self, study_plan: dict):
        """
        Save a study plan to MongoDB.
        If study_plan_id is provided, updates existing document, otherwise creates new one.

        Args:
            study_plan (dict): Study plan data following PlannerOutput schema

        Returns:
            str: The document ID
        """
        # Check if this is an update (has study_plan_id)
        study_plan_id = study_plan.get("study_plan_id")
        if study_plan_id:
            # Update existing document
            if isinstance(study_plan_id, str):
                study_plan_id = ObjectId(study_plan_id)
            # Remove study_plan_id from the document data
            doc_data = {k: v for k, v in study_plan.items() if k != "study_plan_id"}
            result = self.study_plan_collection.replace_one(
                {"_id": study_plan_id}, 
                doc_data,
                upsert=True
            )
            return str(study_plan_id)
        else:
            # Create new document
            result = self.study_plan_collection.insert_one(study_plan)
            return str(result.inserted_id)

    def get_study_plan(self, study_plan_id):
        """Get a study plan by ID."""
        if isinstance(study_plan_id, str):
            study_plan_id = ObjectId(study_plan_id)
        return self.study_plan_collection.find_one({"_id": study_plan_id})

    def save_scheduled_sessions(self, study_plan_id: str, scheduled_sessions: dict):
        """
        Save scheduled sessions to the task_scheduling collection.

        Args:
            study_plan_id (str): The study plan ID these sessions belong to
            scheduled_sessions (dict): The scheduler output containing sessions

        Returns:
            str: The inserted document ID
        """
        doc = {
            "study_plan_id": study_plan_id,
            "course_id": scheduled_sessions.get("course_id"),
            "sessions": scheduled_sessions.get("sessions", []),
            "total_minutes": scheduled_sessions.get("total_minutes", 0),
            "span_days": scheduled_sessions.get("span_days", 1),
            "fallback_used": scheduled_sessions.get("fallback_used", False),
            "skipped_tasks": scheduled_sessions.get("skipped_tasks", []),
            "created_at": datetime.now().isoformat()
        }
        result = self.task_scheduling_collection.insert_one(doc)
        return str(result.inserted_id)
