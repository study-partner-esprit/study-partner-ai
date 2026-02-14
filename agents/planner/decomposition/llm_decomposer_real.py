import requests
from typing import List
import uuid
from agents.planner.models.task_graph import AtomicTask


class LLMDecomposerReal:
    """
    Real LLM decomposer using LM Studio.
    Assumes LM Studio exposes a local REST API.
    """

    def __init__(self, endpoint: str = "http://127.0.0.1:1234/v1/chat/completions"):
        """
        :param endpoint: LM Studio local API URL
        """
        self.endpoint = endpoint

    def decompose(
        self, goal: str, concepts: List[str], available_minutes: int
    ) -> List[AtomicTask]:
        """
        Calls the LLM to generate atomic tasks.
        :param goal: user learning goal
        :param concepts: retrieved concepts from RAG
        :return: list of AtomicTask
        """
        # Build prompt
        context = "\n".join(f"- {c}" for c in concepts)
        prompt = f"""You are a study planner assistant.

Goal: {goal}

Relevant concepts:
{context}

Break this goal into atomic study tasks. Each task should be <= 45 minutes, include review sessions, and respect prerequisites. Total time should fit within {available_minutes} minutes.

Return ONLY a valid JSON array with this exact format:
[{{"title": "task name", "description": "task description", "estimated_minutes": 30, "difficulty": 0.5, "prerequisites": ["prerequisite task title"]}}]

IMPORTANT: Prerequisites must be an array of STRINGS (task titles), not objects!

Example:
[{{"title": "Learn Vector Operations", "description": "Study basic vector addition, scalar multiplication", "estimated_minutes": 30, "difficulty": 0.4, "prerequisites": []}}, {{"title": "Apply Vector Operations", "description": "Practice vector operations with examples", "estimated_minutes": 30, "difficulty": 0.5, "prerequisites": ["Learn Vector Operations"]}}]"""

        # Call LM Studio API
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": "qwen/qwen2.5-vl-7b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.7,
                },
                timeout=120,  # Increased timeout for slower models
            )
            response.raise_for_status()
            result_json = response.json()

            # OpenAI-compatible API returns text in choices[0].message.content
            output_text = (
                result_json.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            # Parse JSON returned by the model
            import json
            import re

            # Extract JSON array from the response text
            json_match = re.search(r"\[.*\]", output_text, re.DOTALL)
            if json_match:
                try:
                    tasks_data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    # If JSON is incomplete, try to extract complete objects
                    # Find all complete task objects
                    task_pattern = r'\{[^{}]*"title"[^{}]*"description"[^{}]*"estimated_minutes"[^{}]*"difficulty"[^{}]*\}'
                    task_matches = re.findall(task_pattern, output_text, re.DOTALL)
                    if task_matches:
                        # Reconstruct valid JSON array
                        fixed_json = "[" + ",".join(task_matches) + "]"
                        tasks_data = json.loads(fixed_json)
                    else:
                        raise ValueError(
                            "Could not extract valid task objects from response"
                        )
            else:
                raise ValueError("No JSON array found in response")

            # Convert to AtomicTask
            tasks = []
            for t in tasks_data:
                # Handle difficulty conversion
                difficulty_str = t.get("difficulty", 0.5)
                if isinstance(difficulty_str, str):
                    difficulty_map = {
                        "beginner": 0.3,
                        "intermediate": 0.5,
                        "advanced": 0.7,
                        "easy": 0.3,
                        "medium": 0.5,
                        "hard": 0.7,
                    }
                    difficulty = difficulty_map.get(difficulty_str.lower(), 0.5)
                else:
                    difficulty = float(difficulty_str)
                
                # Clamp difficulty to valid range (0.0-1.0)
                difficulty = max(0.0, min(1.0, difficulty))

                # Cap estimated_minutes at 45
                estimated_minutes = min(t["estimated_minutes"], 45)

                # Normalize prerequisites to strings (LLM sometimes returns objects)
                prerequisites = t.get("prerequisites", [])
                normalized_prereqs = []
                for prereq in prerequisites:
                    if isinstance(prereq, dict):
                        # Extract title from object: {"title": "Task Name"}
                        normalized_prereqs.append(prereq.get("title", ""))
                    else:
                        # Already a string
                        normalized_prereqs.append(prereq)

                tasks.append(
                    AtomicTask(
                        id=str(uuid.uuid4()),
                        title=t["title"],
                        description=t["description"],
                        estimated_minutes=estimated_minutes,
                        difficulty=difficulty,
                        prerequisites=normalized_prereqs,
                        is_review=False,
                    )
                )

            # Validate and fix prerequisites - only keep prerequisites that exist as task titles
            task_titles = {task.title for task in tasks}
            for task in tasks:
                task.prerequisites = [
                    prereq for prereq in task.prerequisites 
                    if prereq in task_titles
                ]

            return tasks
            
        except Exception as e:
            print(f"[LLMDecomposerReal] Error calling LLM: {e}")
            response_text = (
                response.text if "response" in locals() else "No response available"
            )
            print(f"[LLMDecomposerReal] Response text: {response_text}")
            # Fallback to simple decomposer
            from agents.planner.decomposition.simple_decomposer import (
                SimpleGoalDecomposer,
            )

            simple_decomposer = SimpleGoalDecomposer()
            return simple_decomposer.decompose(goal, concepts, available_minutes)
