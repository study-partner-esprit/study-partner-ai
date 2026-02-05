"""Simple goal decomposer that creates multiple tasks based on goal analysis."""

import uuid
from typing import List
from agents.planner.models.task_graph import AtomicTask


class SimpleGoalDecomposer:
    """
    Simple decomposer that creates multiple meaningful tasks based on goal analysis.

    Analyzes the goal and creates a sequence of learning tasks with appropriate
    difficulty progression and prerequisites.
    """

    def decompose(
        self, goal: str, concepts: List[str] = None, available_minutes: int = 90
    ) -> List[AtomicTask]:
        """
        Decompose goal into multiple atomic tasks.

        Args:
            goal: Learning goal string
            concepts: Retrieved relevant concepts (required from JSON)

        Returns:
            List of AtomicTask objects
        """
        if concepts is None:
            concepts = []

        # Use concepts from JSON documents to create tasks
        return self._create_general_learning_tasks(goal, concepts, available_minutes)

    def _create_general_learning_tasks(
        self, goal: str, concepts: List[str], available_minutes: int
    ) -> List[AtomicTask]:
        """Create general learning tasks based on goal and concepts."""
        tasks = []

        # Create tasks from concepts if available
        if concepts:
            # Calculate number of tasks based on available time (30 min per task)
            num_tasks = min(len(concepts), max(1, available_minutes // 30))
            task_ids = [str(uuid.uuid4()) for _ in range(num_tasks)]
            for i in range(num_tasks):
                concept = concepts[i]  # Get the concept for this task
                prerequisites = [task_ids[i - 1]] if i > 0 else []
                tasks.append(
                    AtomicTask(
                        id=task_ids[i],
                        title=f"Study {concept}",
                        description=f"Learn and understand {concept} as part of {goal}",
                        estimated_minutes=30,
                        difficulty=0.4 + (i * 0.1),  # Increasing difficulty
                        prerequisites=prerequisites,
                    )
                )

        # Add a general study task if no concepts
        if not tasks:
            task1_id = str(uuid.uuid4())
            task2_id = str(uuid.uuid4())
            task3_id = str(uuid.uuid4())

            tasks = [
                AtomicTask(
                    id=task1_id,
                    title=f"Introduction to {goal}",
                    description=f"Basic concepts and overview of {goal}",
                    estimated_minutes=30,
                    difficulty=0.3,
                    prerequisites=[],
                ),
                AtomicTask(
                    id=task2_id,
                    title=f"Core Concepts of {goal}",
                    description=f"Essential principles and foundations of {goal}",
                    estimated_minutes=45,
                    difficulty=0.5,
                    prerequisites=[task1_id],
                ),
                AtomicTask(
                    id=task3_id,
                    title=f"Advanced Topics in {goal}",
                    description=f"Complex aspects and applications of {goal}",
                    estimated_minutes=45,
                    difficulty=0.7,
                    prerequisites=[task2_id],
                ),
            ]

        return tasks
