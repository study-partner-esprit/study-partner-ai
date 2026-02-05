"""Prompt builder for RAG-enhanced task decomposition."""


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
