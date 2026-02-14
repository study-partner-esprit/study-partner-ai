from agents.planner.models.task_graph import (
    PlannerInput,
    PlannerOutput,
    TaskGraph,
)
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

        # Index all course texts
        if course_texts:
            try:
                chunks_added = self.retriever.add_documents(
                    course_texts, request.tokenization_settings
                )
                print(f"✅ Added {chunks_added} document chunks to knowledge base")
            except Exception as e:
                print(f"⚠️ Warning: Could not index documents - {str(e)}")

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
            effective_goal, retrieved_concepts, request.available_minutes, request.course_knowledge
        )

        # Step 6: Apply rules engine
        tasks = self._apply_rules(tasks, request.available_minutes, pacing_factor)

        # Step 7: Create task graph
        task_graph = TaskGraph(goal=effective_goal, tasks=tasks)

        # Step 8: Final feasibility check
        warning = None
        if not is_plan_feasible(tasks, request.available_minutes):
            warning = f"Plan requires {task_graph.total_estimated_minutes} minutes, but only {request.available_minutes} available."

        # Step 9: Update pacing store
        self.pacing_store.update_from_execution(
            request.user_id,
            task_graph.total_estimated_minutes,
            task_graph.total_estimated_minutes,
        )

        return PlannerOutput(
            task_graph=task_graph, warning=warning, clarification_required=False
        )

    def _decompose_goal(
        self, goal: str, concepts: list, available_minutes: int, course_knowledge: dict = None
    ) -> list:
        """
        Decompose the learning goal into atomic tasks.
        If course_knowledge is provided and no specific goal, generate tasks from course structure.
        Tries LLM decomposer first, falls back to simple decomposer.
        """
        # If course knowledge is provided and goal is derived from course, generate tasks from course
        if course_knowledge and goal and self._is_goal_from_course(goal, course_knowledge):
            print(f"DEBUG: Using course-based generation for goal: {goal}")
            return self._generate_tasks_from_course(course_knowledge, available_minutes)

        # Try LLM decomposer first for specific goals
        print(f"DEBUG: Trying LLM decomposer for goal: {goal}")
        llm_tasks = self.llm_decomposer.decompose(goal, concepts, available_minutes)
        if llm_tasks and len(llm_tasks) > 1:
            print(f"DEBUG: LLM decomposer returned {len(llm_tasks)} tasks")
            return llm_tasks

        # Fallback to simple decomposer
        print("Using simple decomposer fallback")
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
        course_title = course_knowledge.get("title") or course_knowledge.get("course_title", "")
        result = course_title and course_title in goal
        print(f"DEBUG: _is_goal_from_course - goal: '{goal}', course_title: '{course_title}', result: {result}")
        return result

    def _generate_tasks_from_course(self, course_knowledge: dict, available_minutes: int) -> list:
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
        course_title = course_knowledge.get("title") or course_knowledge.get("course_title", "Course")

        # Add an introductory task
        intro_task = AtomicTask(
            id=f"task-{task_id_counter:03d}",
            title=f"Course Overview: {course_title}",
            description=f"Review the course structure and learning objectives for {course_title}",
            estimated_minutes=min(30, available_minutes // 4),
            difficulty=0.2,
            prerequisites=[],
            is_review=False
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
                    is_review=False
                )
                tasks.append(topic_task)
                task_id_counter += 1

                # Generate tasks from subtopics
                if "subtopics" in topic:
                    for subtopic_idx, subtopic in enumerate(topic["subtopics"]):
                        subtopic_title = subtopic.get("title", f"Subtopic {subtopic_idx + 1}")

                        # Combine description from available content
                        # Prioritize clean tokenized_chunks over potentially unclean summary
                        description_parts = []
                        if "tokenized_chunks" in subtopic and subtopic["tokenized_chunks"]:
                            # Use the first clean chunk
                            description_parts.append(subtopic["tokenized_chunks"][0][:300])
                            print(f"DEBUG: Using tokenized_chunks for {subtopic_title}: {subtopic['tokenized_chunks'][0][:100]}...")
                        elif "summary" in subtopic:
                            # Fallback to summary if no tokenized chunks
                            description_parts.append(subtopic["summary"][:300])
                            print(f"DEBUG: Using summary for {subtopic_title}: {subtopic['summary'][:100]}...")
                        else:
                            print(f"DEBUG: No content found for {subtopic_title}")

                        description = " ".join(description_parts) if description_parts else f"Study {subtopic_title}"

                        subtopic_task = AtomicTask(
                            id=f"task-{task_id_counter:03d}",
                            title=f"Subtopic: {subtopic_title}",
                            description=description,
                            estimated_minutes=min(30, available_minutes // 8),
                            difficulty=0.6,
                            prerequisites=[topic_task.id],
                            is_review=False
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
                prerequisites=[task.id for task in tasks[-3:]],  # Depend on last few tasks
                is_review=True
            )
            tasks.append(review_task)

        # Scale task times to fit available minutes
        total_time = sum(task.estimated_minutes for task in tasks)
        if total_time > available_minutes:
            scale_factor = available_minutes / total_time
            for task in tasks:
                task.estimated_minutes = max(5, int(task.estimated_minutes * scale_factor))

        return tasks
