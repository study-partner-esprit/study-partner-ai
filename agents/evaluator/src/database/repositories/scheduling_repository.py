"""
Repository for scheduling operations.
"""

import logging
from typing import Dict, Any

from src.database.mongo_client import MongoDBClient
from bson import ObjectId
from bson.errors import InvalidId

logger = logging.getLogger(__name__)


class SchedulingRepository:
    """Repository for scheduling-related operations."""

    def __init__(self, mongo_client: MongoDBClient, collection_name: str = "task_scheduling"):
        self.mongo_client = mongo_client
        self.collection_name = collection_name

    def update_task_status(self, task_id: str, status: str, **kwargs):
        """Update task status."""
        try:
            db = self.mongo_client.db
            collection = db[self.collection_name]
            update_data = {"status": status, **kwargs}
            # Normalize task_id to ObjectId when possible
            try:
                oid = ObjectId(task_id)
            except (InvalidId, TypeError):
                logger.warning(f"Invalid task_id provided, using raw value: {task_id}")
                oid = task_id

            result = collection.update_one({"_id": oid}, {"$set": update_data})
            if result.modified_count > 0:
                logger.info(f"Updated task {task_id} status to {status}")
            else:
                logger.warning(f"No task updated for {task_id}")
        except Exception as e:
            logger.error(f"Error updating task {task_id}: {e}")
            raise
