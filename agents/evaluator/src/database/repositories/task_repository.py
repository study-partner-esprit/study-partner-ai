"""
Repository for task scheduling data from MongoDB.
"""

import logging
from typing import Optional, Dict, Any
from src.database.mongo_client import MongoDBClient

logger = logging.getLogger(__name__)


class TaskRepository:
    """Repository for accessing task scheduling data."""

    def __init__(self, mongo_client: MongoDBClient, collection_name: str = "task_scheduling"):
        self.mongo_client = mongo_client
        self.collection_name = collection_name

    def get_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a task by its ID.
        
        Args:
            task_id: Task ID as a string (UUID string from task_scheduling._id)
            
        Returns:
            Task document or None if not found
        """
        try:
            db = self.mongo_client.db
            if db is None:
                logger.error("Database connection is None")
                return None

            collection = db[self.collection_name]
            if collection is None:
                logger.error(f"Collection {self.collection_name} is None")
                return None

            task = collection.find_one({"_id": task_id})
            if task:
                logger.debug(f"Found task: {task_id}")
            else:
                logger.warning(f"Task not found: {task_id}")
            return task
        except Exception as e:
            logger.error(f"Error fetching task {task_id}: {e}")
            return None

    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a task by its ID (legacy method name)."""
        return self.get_by_id(task_id)
