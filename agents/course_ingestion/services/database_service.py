import os
import traceback
from pymongo import MongoClient
from bson import ObjectId
from agents.course_ingestion.enrichment.llm_enricher import enrich_subtopic_with_llm

# ----------------------------
# MongoDB setup
# ----------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "study_partner")
COLLECTION_NAME = os.getenv("COURSE_COLLECTION", "courses")
STUDY_PLAN_COLLECTION = os.getenv("STUDY_PLAN_COLLECTION", "study_plans")

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
    
    def save_course(self, course: dict):
        """
        Enrich all subtopics with LLM before saving to MongoDB.

        Args:
            course (dict): Course JSON following CourseKnowledgeJSON schema

        Returns:
            str: The inserted document ID
        """
        topics = course.get("topics", [])

        for topic in topics:
            subtopics = topic.get("subtopics", [])

            for sub in subtopics:
                # Use tokenized chunks if available, otherwise summary
                text = "\n".join(sub.get("tokenized_chunks", [])) or sub.get("summary", "")

                try:
                    enriched = enrich_subtopic_with_llm(
                        title=sub.get("title", ""),
                        text=text
                    )

                    # Overwrite fields with enriched content
                    sub["summary"] = enriched.get("cleaned_text", sub.get("summary", ""))
                    sub["key_concepts"] = enriched.get("key_concepts", [])
                    sub["definitions"] = enriched.get("definitions", [])
                    sub["formulas"] = enriched.get("formulas", [])
                    sub["examples"] = enriched.get("examples", [])

                except Exception:
                    # fallback: keep original content, log error
                    print(f"⚠️ LLM enrichment failed for subtopic {sub.get('id')}")
                    traceback.print_exc()
                    continue

        # Save final enriched course to MongoDB
        result = self.collection.insert_one(course)
        return str(result.inserted_id)
    
    def get_course(self, course_id):
        """Get a course by ID."""
        if isinstance(course_id, str):
            course_id = ObjectId(course_id)
        return self.collection.find_one({"_id": course_id})
    
    def save_study_plan(self, study_plan: dict):
        """
        Save a study plan to MongoDB.

        Args:
            study_plan (dict): Study plan data following PlannerOutput schema

        Returns:
            str: The inserted document ID
        """
        result = self.study_plan_collection.insert_one(study_plan)
        return str(result.inserted_id)
    
    def get_study_plan(self, study_plan_id):
        """Get a study plan by ID."""
        if isinstance(study_plan_id, str):
            study_plan_id = ObjectId(study_plan_id)
        return self.study_plan_collection.find_one({"_id": study_plan_id})
    
    def get_study_plans_by_user(self, user_id: str):
        """Get all study plans for a user."""
        return list(self.study_plan_collection.find({"task_graph.user_id": user_id}))
