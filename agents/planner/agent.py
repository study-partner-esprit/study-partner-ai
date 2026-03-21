import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from agents.planner.models.task_graph import (
    PlannerInput,
    PlannerOutput,
    TaskGraph,
)
from models.task import Task
from agents.planner.decomposition.simple_decomposer import SimpleGoalDecomposer
from agents.planner.decomposition.llm_decomposer_real import LLMDecomposerReal
from agents.planner.rules.constraints import enforce_max_duration
from agents.planner.rules.review_inserter import insert_review_tasks
from agents.planner.rules.buffer_inserter import insert_buffers
from agents.planner.rules.feasibility import is_plan_feasible
from agents.planner.rules.clarification import ClarificationChecker
from agents.planner.memory.pacing_store import PacingStore
from agents.planner.rag.index_store import (
    save_index,
    load_index,
    load_embeddings,
    rebuild_index_from_embeddings,
)

# RAG + embeddings imports
from agents.planner.rag.embeddings import EmbeddingModel
from agents.planner.rag.indexer import VectorStore
from agents.planner.rag.retriever import ContentRetriever
from utils.logger import get_logger

logger = get_logger(__name__)


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
            request: PlannerInput containing goal/course, deadline, available time, etc.

        Returns:
            PlannerOutput with task graph and any warnings
        """
        # Determine the effective goal
        effective_goal = request.goal
        if not effective_goal and request.course_knowledge:
            # Derive goal from course title
            effective_goal = self._derive_goal_from_course(request.course_knowledge)

        # Step 1: Check goal clarity (if goal provided)
        if effective_goal and self.clarifier.check_goal(effective_goal):
            return PlannerOutput(
                task_graph=TaskGraph(goal=effective_goal, tasks=[]),
                warning="Goal is too vague. Please provide more specific details about what you want to learn.",
                clarification_required=True,
            )

        # Step 2: Index course documents if provided
        course_texts = []

        # Extract text from course_documents (legacy support)
        if request.course_documents:
            course_texts.extend(request.course_documents)

        # Extract text from course_knowledge (new structured format)
        if request.course_knowledge:
            course_texts.extend(
                self._extract_texts_from_course_knowledge(request.course_knowledge)
            )

        # Index all course texts — prefer pre-computed embeddings when available
        if course_texts:
            try:
                chunks_added = self._load_retrieval_context(
                    request.course_knowledge,
                    course_texts,
                    request.tokenization_settings,
                )
                logger.info(
                    "planner_chunks_indexed",
                    extra={"num_chunks": chunks_added},
                )
            except Exception as e:
                logger.warning("planner_index_failed", extra={"error": str(e)})

        # Step 3: Retrieve relevant concepts using RAG
        if course_texts:
            retrieved_concepts = request.retrieved_concepts or self.retriever.retrieve(
                effective_goal, top_k=8
            )
        else:
            retrieved_concepts = []

        # Step 4: Adjust pacing based on user history
        pacing_factor = self.pacing_store.get_user_pacing_factor(request.user_id)

        # Step 5: Decompose goal/course into tasks
        tasks = self._decompose_goal(
            effective_goal,
            retrieved_concepts,
            request.available_minutes,
            request.course_knowledge,
        )

        # Step 6: Apply rules engine
        tasks = self._apply_rules(tasks, request.available_minutes, pacing_factor)

        # Step 7: Create task graph
        task_graph = TaskGraph(goal=effective_goal, tasks=tasks)

        # Step 8: Final feasibility check
        warning = None
        if not is_plan_feasible(tasks, request.available_minutes):
            warning = (
                f"Plan requires {task_graph.total_estimated_minutes} minutes, "
                f"but only {request.available_minutes} available."
            )

        # Step 9: Persist FAISS index to disk if we built one
        course_id = (
            request.course_knowledge.get("_id")
            or request.course_knowledge.get("course_title", "default")
            if request.course_knowledge
            else None
        )
        if course_id and self.retriever.indexed_chunks:
            try:
                save_index(
                    self.vector_store.index,
                    self.retriever.indexed_chunks,
                    str(course_id),
                    embeddings=getattr(self.retriever, "last_embeddings", None),
                )
            except Exception as exc:
                logger.warning("planner_save_index_failed", extra={"error": str(exc)})

        # Step 10: Update pacing store
        self.pacing_store.update_from_execution(
            request.user_id,
            task_graph.total_estimated_minutes,
            task_graph.total_estimated_minutes,
        )

        return PlannerOutput(
            task_graph=task_graph, warning=warning, clarification_required=False
        )

    # ------------------------------------------------------------------
    # Convenience wrapper — maps kwargs to PlannerInput + converts output
    # ------------------------------------------------------------------

    def generate_tasks(
        self,
        user_goal: str,
        available_time_minutes: int,
        course_context: Optional[Dict[str, Any]] = None,
        user_id: str = "",
        deadline_days: int = 7,
    ) -> List[Task]:
        """
        Convenience method: build a list of scheduler-ready ``Task`` objects
        from a natural-language goal (and optional course context).

        Args:
            user_goal:              The learner's main goal.
            available_time_minutes: Total study budget in minutes.
            course_context:         Optional structured course knowledge dict.
            user_id:                User identifier (defaults to a random UUID).
            deadline_days:          Days from now until the study deadline.

        Returns:
            List[Task] ready to pass to ``SchedulerAgent.build_schedule()``.
        """
        if not user_id:
            user_id = str(uuid.uuid4())

        deadline = (datetime.now() + timedelta(days=deadline_days)).isoformat()

        request = PlannerInput(
            goal=user_goal,
            deadline_iso=deadline,
            available_minutes=available_time_minutes,
            user_id=user_id,
            course_knowledge=course_context,
        )

        output: PlannerOutput = self.plan(request)
        atomic_tasks = output.task_graph.tasks

        def _difficulty_label(score: float) -> str:
            if score < 0.35:
                return "easy"
            if score < 0.65:
                return "medium"
            return "hard"

        return [
            Task(
                task_id=at.id,
                user_id=user_id,
                title=at.title,
                description=at.description,
                estimated_duration=at.estimated_minutes,
                difficulty=_difficulty_label(at.difficulty),
                prerequisites=at.prerequisites,
                status="pending",
            )
            for at in atomic_tasks
        ]

    def _load_retrieval_context(
        self,
        course_knowledge: dict | None,
        course_texts: list[str],
        tokenization_settings: dict | None,
    ) -> int:
        """
        Populate the in-memory FAISS retriever, preferring pre-computed embeddings.

        Priority:
          1. Pre-computed chunk_embeddings from course_knowledge subtopics.
          2. Disk-cached FAISS index (saves re-encoding on repeated calls).
          3. Re-encode from raw course_texts (slowest path).

        Returns:
            Number of chunks indexed.
        """
        # --- path 1: pre-computed embeddings from MongoDB / ingestion --- #
        if course_knowledge and "topics" in course_knowledge:
            all_chunks: list[str] = []
            all_embeddings: list[list[float]] = []
            for topic in course_knowledge["topics"]:
                for subtopic in topic.get("subtopics", []):
                    chunks = subtopic.get("tokenized_chunks", [])
                    embeds = subtopic.get("chunk_embeddings")
                    if chunks and embeds and len(chunks) == len(embeds):
                        all_chunks.extend(chunks)
                        all_embeddings.extend(embeds)
            if all_chunks:
                logger.info(
                    "planner_using_precomputed_embeddings",
                    extra={"num_chunks": len(all_chunks)},
                )
                return self.retriever.add_precomputed_embeddings(
                    all_chunks, all_embeddings
                )

        # --- path 2: disk-cached FAISS index --- #
        course_id = None
        if course_knowledge:
            course_id = str(
                course_knowledge.get("_id") or course_knowledge.get("course_title", "")
            )
        if course_id:
            disk_index, disk_chunks = load_index(course_id)
            if disk_index is not None:
                logger.info(
                    "planner_loaded_disk_index",
                    extra={"course_id": course_id, "ntotal": disk_index.ntotal},
                )
                self.vector_store.index = disk_index
                self.retriever.indexed_chunks = disk_chunks
                return len(disk_chunks)
            # Recovery path: rebuild index from stored embeddings when index files are missing/corrupt.
            disk_embeddings = load_embeddings(course_id)
            if disk_embeddings is not None and len(course_texts) == len(disk_embeddings):
                rebuilt_index, rebuilt_chunks = rebuild_index_from_embeddings(
                    course_id, course_texts, disk_embeddings
                )
                if rebuilt_index is not None:
                    self.vector_store.index = rebuilt_index
                    self.retriever.indexed_chunks = rebuilt_chunks or []
                    logger.info(
                        "planner_rebuilt_disk_index",
                        extra={
                            "course_id": course_id,
                            "ntotal": rebuilt_index.ntotal,
                        },
                    )
                    return len(self.retriever.indexed_chunks)

        # --- path 3: fallback — re-encode from text --- #
        logger.info(
            "planner_reencoding_course_texts", extra={"num_texts": len(course_texts)}
        )
        return self.retriever.add_documents(course_texts, tokenization_settings)

    def _decompose_goal(
        self,
        goal: str,
        concepts: list,
        available_minutes: int,
        course_knowledge: dict = None,
    ) -> list:
        """
        Decompose the learning goal into atomic tasks.
        If course_knowledge is provided and no specific goal, generate tasks from course structure.
        Tries LLM decomposer first, falls back to simple decomposer.
        """
        # If course knowledge is provided and goal is derived from course, generate tasks from course
        if (
            course_knowledge
            and goal
            and self._is_goal_from_course(goal, course_knowledge)
        ):
            logger.debug("planner_course_based_decompose", extra={"goal": goal})
            return self._generate_tasks_from_course(course_knowledge, available_minutes)

        # Try LLM decomposer first for specific goals
        logger.debug("planner_llm_decompose", extra={"goal": goal})
        llm_tasks = self.llm_decomposer.decompose(goal, concepts, available_minutes)
        if llm_tasks and len(llm_tasks) > 1:
            logger.debug(
                "planner_llm_decompose_ok", extra={"num_tasks": len(llm_tasks)}
            )
            return llm_tasks

        # Fallback to simple decomposer
        logger.info("planner_simple_decompose_fallback")
        return self.simple_decomposer.decompose(goal, concepts, available_minutes)

    def _apply_rules(
        self, tasks: list, available_minutes: int, pacing_factor: float
    ) -> list:
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
        tasks = insert_review_tasks(tasks)  # Add review sessions
        tasks = insert_buffers(tasks)  # Add buffer time

        return tasks

    def _extract_texts_from_course_knowledge(self, course_knowledge: dict) -> list[str]:
        """
        Extract text content from structured course knowledge JSON.

        Args:
            course_knowledge: Normalized course JSON from ingestion agent

        Returns:
            List of text chunks for RAG indexing
        """
        texts = []

        # Extract course title
        if "title" in course_knowledge:
            texts.append(f"Course Title: {course_knowledge['title']}")
        elif "course_title" in course_knowledge:
            texts.append(f"Course Title: {course_knowledge['course_title']}")

        # Extract topic and subtopic content
        if "topics" in course_knowledge:
            for topic in course_knowledge["topics"]:
                if "title" in topic:
                    texts.append(f"Topic: {topic['title']}")

                if "subtopics" in topic:
                    for subtopic in topic["subtopics"]:
                        # Combine all available text content
                        text_parts = []

                        if "title" in subtopic:
                            text_parts.append(f"Subtopic: {subtopic['title']}")

                        if "summary" in subtopic:
                            text_parts.append(subtopic["summary"])

                        if "tokenized_chunks" in subtopic:
                            text_parts.extend(subtopic["tokenized_chunks"])

                        if text_parts:
                            combined_text = " ".join(text_parts)
                            if combined_text.strip():
                                texts.append(combined_text)

        return texts

    def _derive_goal_from_course(self, course_knowledge: dict) -> str:
        """
        Derive a learning goal from course knowledge.

        Args:
            course_knowledge: Structured course data

        Returns:
            Derived learning goal string
        """
        if "title" in course_knowledge:
            return f"Complete the course: {course_knowledge['title']}"
        elif "course_title" in course_knowledge:
            return f"Complete the course: {course_knowledge['course_title']}"
        else:
            return "Complete the provided course materials"

    def _is_goal_from_course(self, goal: str, course_knowledge: dict) -> bool:
        """
        Check if the goal was derived from the course knowledge.

        Args:
            goal: The learning goal
            course_knowledge: Structured course data

        Returns:
            True if goal appears to be derived from course
        """
        course_title = course_knowledge.get("title") or course_knowledge.get(
            "course_title", ""
        )
        result = bool(course_title and course_title in goal)
        logger.debug(
            "planner_is_goal_from_course",
            extra={"goal": goal, "course_title": course_title, "result": result},
        )
        return result

    def _generate_tasks_from_course(
        self, course_knowledge: dict, available_minutes: int
    ) -> list:
        """
        Generate tasks directly from course structure.

        Args:
            course_knowledge: Structured course data
            available_minutes: Total available time

        Returns:
            List of AtomicTask objects covering the course
        """
        from agents.planner.models.task_graph import AtomicTask
        import uuid

        tasks = []
        task_id_counter = 1

        # Extract course title
        course_title = course_knowledge.get("title") or course_knowledge.get(
            "course_title", "Course"
        )

        # Add an introductory task
        intro_task = AtomicTask(
            id=f"task-{task_id_counter:03d}",
            title=f"Course Overview: {course_title}",
            description=f"Review the course structure and learning objectives for {course_title}",
            estimated_minutes=min(30, available_minutes // 4),
            difficulty=0.2,
            prerequisites=[],
            is_review=False,
        )
        tasks.append(intro_task)
        task_id_counter += 1

        # Generate tasks from topics and subtopics
        if "topics" in course_knowledge:
            for topic_idx, topic in enumerate(course_knowledge["topics"]):
                topic_title = topic.get("title", f"Topic {topic_idx + 1}")

                # Add topic-level task
                topic_task = AtomicTask(
                    id=f"task-{task_id_counter:03d}",
                    title=f"Topic: {topic_title}",
                    description=f"Study the main concepts in {topic_title}",
                    estimated_minutes=min(45, available_minutes // 6),
                    difficulty=0.5,
                    prerequisites=[intro_task.id] if tasks else [],
                    is_review=False,
                )
                tasks.append(topic_task)
                task_id_counter += 1

                # Generate tasks from subtopics
                if "subtopics" in topic:
                    for subtopic_idx, subtopic in enumerate(topic["subtopics"]):
                        subtopic_title = subtopic.get(
                            "title", f"Subtopic {subtopic_idx + 1}"
                        )

                        # Combine description from available content
                        # Prioritize clean tokenized_chunks over potentially unclean summary
                        description_parts = []
                        if (
                            "tokenized_chunks" in subtopic
                            and subtopic["tokenized_chunks"]
                        ):
                            # Use the first clean chunk
                            description_parts.append(
                                subtopic["tokenized_chunks"][0][:300]
                            )
                        elif "summary" in subtopic:
                            # Fallback to summary if no tokenized chunks
                            description_parts.append(subtopic["summary"][:300])

                        description = (
                            " ".join(description_parts)
                            if description_parts
                            else f"Study {subtopic_title}"
                        )

                        subtopic_task = AtomicTask(
                            id=f"task-{task_id_counter:03d}",
                            title=f"Subtopic: {subtopic_title}",
                            description=description,
                            estimated_minutes=min(30, available_minutes // 8),
                            difficulty=0.6,
                            prerequisites=[topic_task.id],
                            is_review=False,
                        )
                        tasks.append(subtopic_task)
                        task_id_counter += 1

        # Add review task at the end
        if tasks:
            review_task = AtomicTask(
                id=f"task-{task_id_counter:03d}",
                title=f"Course Review: {course_title}",
                description=f"Review all key concepts from {course_title}",
                estimated_minutes=min(45, available_minutes // 4),
                difficulty=0.4,
                prerequisites=[
                    task.id for task in tasks[-3:]
                ],  # Depend on last few tasks
                is_review=True,
            )
            tasks.append(review_task)

        # Scale task times to fit available minutes
        total_time = sum(task.estimated_minutes for task in tasks)
        if total_time > available_minutes:
            scale_factor = available_minutes / total_time
            for task in tasks:
                task.estimated_minutes = max(
                    5, int(task.estimated_minutes * scale_factor)
                )

        return tasks
