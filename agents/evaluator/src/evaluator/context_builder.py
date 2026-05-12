"""
Context builder for evaluation tasks from MongoDB data.
"""

import logging
from typing import Optional, Dict, Any, List

from src.database.repositories.course_repository import CourseRepository
from src.database.repositories.task_repository import TaskRepository
from src.evaluator.schemas import TaskEvaluationContext, SubTopic

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds evaluation context from MongoDB data."""

    def __init__(self, course_repo: CourseRepository, task_repo: Optional[TaskRepository] = None):
        self.course_repo = course_repo
        self.task_repo = task_repo

    def build_evaluation_context(self, task_id: str) -> TaskEvaluationContext:
        """
        Build evaluation context for a task from MongoDB.

        Flow:
        1. Fetch task from task_scheduling (using task_id string directly)
        2. Extract course_id from task (ObjectId string)
        3. Fetch course from courses (converting course_id string to ObjectId)
        4. Flatten topics → subtopics
        5. Aggregate key_concepts and deduplicate

        Args:
            task_id: Task ID as a string (UUID string from task_scheduling._id)

        Returns:
            TaskEvaluationContext with real data from MongoDB

        Raises:
            ValueError: If task not found or course not found
        """
        # Fetch task from task_scheduling
        if self.task_repo is None:
            msg = "TaskRepository required to build evaluation context"
            logger.error(msg)
            raise ValueError(msg)

        task_data = self.task_repo.get_by_id(task_id)
        if task_data is None:
            msg = f"Task not found: {task_id}"
            logger.error(msg)
            raise ValueError(msg)

        # Extract course_id from task (stored as string)
        course_id = task_data.get("course_id")
        if course_id is None:
            msg = f"Task {task_id} missing course_id field"
            logger.error(msg)
            raise ValueError(msg)

        logger.debug(f"Task {task_id} references course {course_id}")

        # Fetch course from courses collection
        course = self.course_repo.get_by_id(course_id)
        if course is None:
            msg = f"Course not found: {course_id}"
            logger.error(msg)
            raise ValueError(msg)

        logger.debug(f"Fetched course {course_id}: {course.get('title', 'Unknown')}")

        # Extract course title
        course_title = course.get("title", "Unknown Course")

        # Flatten topics → subtopics and aggregate key_concepts
        subtopics = self._flatten_subtopics(course)
        key_concepts_set = self._aggregate_key_concepts(course)

        context = TaskEvaluationContext(
            task_id=task_id,
            course_id=course_id,
            course_title=course_title,
            subtopics=subtopics,
            key_concepts=list(key_concepts_set),
        )

        logger.info(f"Built evaluation context for task {task_id}")
        logger.debug(
            f"Context: {len(subtopics)} subtopics, {len(key_concepts_set)} unique concepts"
        )
        return context

    def _flatten_subtopics(self, course: Dict[str, Any]) -> List[SubTopic]:
        """
        Flatten topics → subtopics structure from course.

        Current Mongo structure:
        course
         └── topics[]
              └── subtopics[]
                   └── key_concepts[]

        Args:
            course: Course document from MongoDB

        Returns:
            Flattened list of SubTopic objects
        """
        subtopics = []
        topics = course.get("topics", [])

        if not isinstance(topics, list):
            logger.warning(f"Course topics is not a list: {type(topics)}")
            return subtopics

        for topic in topics:
            if not isinstance(topic, dict):
                logger.warning(f"Topic is not a dict: {topic}")
                continue

            topic_subtopics = topic.get("subtopics", [])
            if not isinstance(topic_subtopics, list):
                logger.warning(f"Subtopics for topic {topic.get('title')} is not a list")
                continue

            for sub in topic_subtopics:
                if isinstance(sub, str):
                    # Simple string subtopic
                    sub_obj = SubTopic(
                        id=sub,
                        title=sub,
                        summary=None,
                        key_concepts=[]
                    )
                elif isinstance(sub, dict):
                    # Dict subtopic with nested structure
                    sub_obj = SubTopic(
                        id=sub.get("id", sub.get("title", "unknown")),
                        title=sub.get("title", "Unknown"),
                        summary=sub.get("summary"),
                        key_concepts=sub.get("key_concepts", [])
                    )
                else:
                    logger.warning(f"Subtopic has unexpected type: {type(sub)}")
                    continue

                subtopics.append(sub_obj)

        return subtopics

    def _aggregate_key_concepts(self, course: Dict[str, Any]) -> set:
        """
        Aggregate all key_concepts from topics → subtopics.
        Deduplicates by converting to a set.

        Args:
            course: Course document from MongoDB

        Returns:
            Set of unique key concepts
        """
        concepts_set = set()

        # Top-level course concepts
        course_concepts = course.get("key_concepts", [])
        if isinstance(course_concepts, list):
            concepts_set.update(course_concepts)

        # Concepts from topics → subtopics
        topics = course.get("topics", [])
        if isinstance(topics, list):
            for topic in topics:
                if isinstance(topic, dict):
                    subtopics = topic.get("subtopics", [])
                    if isinstance(subtopics, list):
                        for sub in subtopics:
                            if isinstance(sub, dict):
                                sub_concepts = sub.get("key_concepts", [])
                                if isinstance(sub_concepts, list):
                                    concepts_set.update(sub_concepts)

        logger.debug(f"Aggregated {len(concepts_set)} unique key concepts")
        return concepts_set
