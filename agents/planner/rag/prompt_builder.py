"""Prompt builder for RAG-enhanced task decomposition."""

import os
from pymongo import MongoClient
from datetime import datetime
from typing import Optional


class SchedulingService:
    """Service for saving and retrieving study plans from MongoDB."""
    
    def __init__(self):
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("MONGO_DB_NAME", "study_partner")
        
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.study_plan_collection = self.db["study_plans"]
        self.task_scheduling_collection = self.db["task_scheduling"]
    
    def save_study_plan(self, user_id: str, study_plan: dict) -> str:
        """Save a study plan and create task scheduling."""
        # Add metadata
        study_plan["user_id"] = user_id
        study_plan["created_at"] = datetime.now()
        
        # Save study plan
        result = self.study_plan_collection.insert_one(study_plan)
        study_plan_id = str(result.inserted_id)
        
        # Extract task graph and create scheduling
        task_graph = study_plan.get("task_graph", {})
        atomic_tasks = task_graph.get("atomic_tasks", [])
        
        # Create sessions from atomic tasks
        sessions = []
        current_time = study_plan.get("start_date", datetime.now())
        
        for task in atomic_tasks:
            session = {
                "task_id": task["id"],
                "start_datetime": current_time,
                "end_datetime": current_time,  # TODO: Calculate based on duration
                "duration_minutes": task.get("estimated_minutes", 30)
            }
            sessions.append(session)
        
        # Save scheduling
        scheduling_doc = {
            "user_id": user_id,
            "study_plan_id": study_plan_id,
            "sessions": sessions,
            "status": "active",
            "created_at": datetime.now()
        }
        self.task_scheduling_collection.insert_one(scheduling_doc)
        
        return study_plan_id
    
    def get_user_plans(self, user_id: str, limit: int = 10):
        """Get study plans for a user."""
        plans = self.study_plan_collection.find(
            {"user_id": user_id},
            sort=[("created_at", -1)],
            limit=limit
        )
        return list(plans)


class PromptBuilder:
    """Builds prompts for LLM-based task decomposition with RAG context."""

    def __init__(self):
        pass

    def build_decomposition_prompt(
        self, goal: str, concepts: list, available_minutes: int
    ) -> str:
        """Build a prompt for task decomposition."""
        context = "\n".join(f"- {c}" for c in concepts)
        prompt = f"""You are a study planner assistant.

Goal: {goal}

Relevant concepts:
{context}

Break this goal into atomic study tasks. Each task should be <= 45 minutes, include review sessions, and respect prerequisites. Total time should fit within {available_minutes} minutes.

Return ONLY a valid JSON array with this exact format:
[{{"title": "task name", "description": "task description", "estimated_minutes": 30, "difficulty": 0.5, "prerequisites": []}}]

Example:
[{{"title": "Learn Vector Operations", "description": "Study basic vector addition, scalar multiplication", "estimated_minutes": 30, "difficulty": 0.4, "prerequisites": []}}]"""

        return prompt
