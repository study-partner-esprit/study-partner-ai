"""
Repository for course data from MongoDB.
"""

import logging
from typing import List, Dict, Any, Optional

from src.database.mongo_client import MongoDBClient
from bson import ObjectId
from bson.errors import InvalidId

logger = logging.getLogger(__name__)


class CourseRepository:
    """Repository for accessing course data."""

    def __init__(self, mongo_client: MongoDBClient, collection_name: str = "courses"):
        self.mongo_client = mongo_client
        self.collection_name = collection_name

    def get_by_id(self, course_id: str) -> Optional[Dict[str, Any]]:
        """Get a course by its ID.
        
        Args:
            course_id: Course ID as a string (ObjectId string from task_scheduling.course_id)
            
        Returns:
            Course document or None if not found
        """
        try:
            object_id = ObjectId(course_id)
        except InvalidId:
            logger.warning(f"Invalid course_id format: {course_id}")
            return None

        try:
            db = self.mongo_client.db
            if db is None:
                logger.error("Database connection is None")
                return None

            collection = db[self.collection_name]
            if collection is None:
                logger.error(f"Collection {self.collection_name} is None")
                return None

            course = collection.find_one({"_id": object_id})
            if course:
                logger.debug(f"Found course: {course_id}")
            else:
                logger.warning(f"Course not found: {course_id}")
            return course
        except Exception as e:
            logger.error(f"Error fetching course {course_id}: {e}")
            return None

    def get_course_by_id(self, course_id: str) -> Optional[Dict[str, Any]]:
        """Get a course by its ID (legacy method name)."""
        return self.get_by_id(course_id)

    def get_all_courses(self) -> List[Dict[str, Any]]:
        """Get all courses."""
        try:
            db = self.mongo_client.db
            if db is None:
                logger.error("Database connection is None")
                return []

            collection = db[self.collection_name]
            if collection is None:
                logger.error(f"Collection {self.collection_name} is None")
                return []

            courses = list(collection.find({}))
            logger.debug(f"Found {len(courses)} courses")
            return courses
        except Exception as e:
            logger.error(f"Error fetching all courses: {e}")
            raise

    def get_topic_by_name(self, course_id: str, topic_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific topic from a course."""
        try:
            course = self.get_by_id(course_id)
            if not course:
                return None

            # Assuming topics are in a list or dict
            topics = course.get("topics", [])
            if isinstance(topics, list):
                for topic in topics:
                    if topic.get("name") == topic_name:
                        return topic
            elif isinstance(topics, dict):
                return topics.get(topic_name)

            logger.warning(f"Topic {topic_name} not found in course {course_id}")
            return None
        except Exception as e:
            logger.error(f"Error fetching topic {topic_name} from course {course_id}: {e}")
            raise