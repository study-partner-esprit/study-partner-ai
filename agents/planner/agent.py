from agents.planner.models.task_graph import PlannerInput, PlannerOutput, TaskGraph, AtomicTask
from agents.planner.decomposition.simple_decomposer import SimpleGoalDecomposer
from agents.planner.decomposition.llm_decomposer_real import LLMDecomposerReal
from agents.planner.rules.constraints import enforce_max_duration
from agents.planner.rules.review_inserter import insert_review_tasks
from agents.planner.rules.buffer_inserter import insert_buffers
from agents.planner.rules.feasibility import is_plan_feasible
from agents.planner.rules.clarification import ClarificationChecker
from agents.planner.memory.pacing_store import PacingStore

# RAG + embeddings imports
from agents.planner.rag.embeddings import EmbeddingModel
from agents.planner.rag.indexer import VectorStore
from agents.planner.rag.retriever import ContentRetriever

import uuid
from datetime import datetime, timedelta


class PlannerAgent:
    """
    Main planner agent that decomposes learning goals into atomic tasks.

    Uses a combination of RAG, LLM, and rules engine to create personalized
    study plans with proper time management and difficulty assessment.
    """

    def __init__(self):
        """Initialize all components of the planner agent."""
        # Decomposition components
        self.simple_decomposer = SimpleGoalDecomposer()
        self.llm_decomposer = LLMDecomposerReal()

        # Rules engine components
        self.clarifier = ClarificationChecker()

        # RAG components
        self.embed_model = EmbeddingModel()
        self.vector_store = VectorStore(dim=384)  # SentenceTransformers default dim
        self.retriever = ContentRetriever(self.vector_store, self.embed_model)

        # Memory component
        self.pacing_store = PacingStore()

        # Note: Knowledge base starts empty and is populated from course_documents

    def plan(self, request: PlannerInput) -> PlannerOutput:
        """
        Main planning method that orchestrates the entire planning process.

        Args:
            request: PlannerInput containing goal, deadline, available time, etc.

        Returns:
            PlannerOutput with task graph and any warnings
        """
        # Step 1: Check goal clarity
        if self.clarifier.check_goal(request.goal):
            return PlannerOutput(
                task_graph=TaskGraph(goal=request.goal, tasks=[]),
                warning="Goal is too vague. Please provide more specific details about what you want to learn.",
                clarification_required=True
            )

        # Step 2: Index course documents if provided
        if request.course_documents:
            try:
                chunks_added = self.retriever.add_documents(
                    request.course_documents,
                    request.tokenization_settings
                )
                print(f"✅ Added {chunks_added} document chunks to knowledge base")
            except Exception as e:
                print(f"⚠️ Warning: Could not index documents - {str(e)}")

        # Step 3: Retrieve relevant concepts using RAG
        if request.course_documents:
            retrieved_concepts = request.retrieved_concepts or self.retriever.retrieve(request.goal, top_k=8)
        else:
            retrieved_concepts = []

        # Step 4: Adjust pacing based on user history
        pacing_factor = self.pacing_store.get_user_pacing_factor(request.user_id)

        # Step 5: Decompose goal into tasks
        tasks = self._decompose_goal(request.goal, retrieved_concepts, request.available_minutes)

        # Step 6: Apply rules engine
        tasks = self._apply_rules(tasks, request.available_minutes, pacing_factor)

        # Step 7: Create task graph
        task_graph = TaskGraph(goal=request.goal, tasks=tasks)

        # Step 8: Final feasibility check
        warning = None
        if not is_plan_feasible(tasks, request.available_minutes):
            warning = f"Plan requires {task_graph.total_estimated_minutes} minutes, but only {request.available_minutes} available."

        # Step 9: Update pacing store
        self.pacing_store.update_from_execution(
            request.user_id,
            task_graph.total_estimated_minutes,
            task_graph.total_estimated_minutes
        )

        return PlannerOutput(
            task_graph=task_graph,
            warning=warning,
            clarification_required=False
        )

    def _decompose_goal(self, goal: str, concepts: list, available_minutes: int) -> list:
        """
        Decompose the learning goal into atomic tasks.
        Tries LLM decomposer first, falls back to simple decomposer.
        """
        # Try LLM decomposer first
        llm_tasks = self.llm_decomposer.decompose(goal, concepts, available_minutes)
        if llm_tasks and len(llm_tasks) > 1:
            return llm_tasks

        # Fallback to simple decomposer
        print("Using simple decomposer fallback")
        return self.simple_decomposer.decompose(goal, concepts, available_minutes)

    def _apply_rules(self, tasks: list, available_minutes: int, pacing_factor: float) -> list:
        """
        Apply all planning rules to the task list.

        Args:
            tasks: List of AtomicTask objects
            available_minutes: Total available time
            pacing_factor: User's pacing adjustment factor

        Returns:
            Modified list of tasks with rules applied
        """
        # Adjust task durations based on pacing
        for task in tasks:
            task.estimated_minutes = int(task.estimated_minutes * pacing_factor)

        # Apply rules in order
        tasks = enforce_max_duration(tasks)  # Max 45 minutes per task
        tasks = insert_review_tasks(tasks)   # Add review sessions
        tasks = insert_buffers(tasks)        # Add buffer time

        return tasks