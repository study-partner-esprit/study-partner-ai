"""
Main EvaluatorAgent orchestrator for standalone Gradio-based Socratic Evaluator.
Handles evaluation pipeline with Google Gemini API, no database required.
Minimizes API usage: only calls Gemini for first question and final evaluation.
"""

import logging
from typing import Optional, Dict, Any
import uuid
from datetime import datetime

from agents.course_ingestion.services.database_service import DatabaseService

from src.evaluator.llm_client import GeminiClient
from src.evaluator.prompts import (
    build_question_prompt,
    build_analysis_prompt,
    generate_followup_question,
    generate_template_question,
    validate_question,
    has_generic_question_terms,
    question_contains_concept,
    concept_coverage,
)
from src.evaluator.schemas import (
    LLMAnalysisResponse,
    TaskEvaluationContext,
    EvaluationSession,
    SessionState,
    EvaluationState,
    RewardPayload,
    ReschedulePayload,
    EvaluationResult,
)
from src.evaluator.scoring import MasteryScorer
from src.evaluator.reward_engine import RewardEngine

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """Main Socratic evaluation orchestrator."""
    
    def __init__(self, llm_client: Optional[GeminiClient] = None, require_llm: bool = True):
        """
        Initialize evaluator agent.
        
        Args:
            llm_client: Optional pre-initialized GeminiClient. If None and require_llm=True, creates new instance.
            require_llm: If True (default), requires GeminiClient for Socratic evaluation. 
                         If False, allows initialization without LLM for post-session evaluation only.
        
        Examples:
            # Use default Gemini client (requires GEMINI_API_KEY)
            agent = EvaluatorAgent()
            
            # Use custom Gemini client
            from src.evaluator.llm_client import GeminiClient
            agent = EvaluatorAgent(llm_client=GeminiClient())
            
            # Post-session evaluation only (no LLM needed)
            agent = EvaluatorAgent(require_llm=False)
        """
        try:
            if llm_client is not None:
                self.llm = llm_client
                logger.info(f"✓ Using provided GeminiClient for evaluation")
            elif require_llm:
                self.llm = GeminiClient()
                logger.info(f"✓ Using GeminiClient for evaluation")
            else:
                self.llm = None
                logger.info(f"✓ Initialized without LLM (post-session evaluation mode)")
            
            # Initialize session storage and scoring
            self.db = DatabaseService()
            self.sessions: Dict[str, EvaluationSession] = {}
            self.scorer = MasteryScorer()
            self.reward_engine = RewardEngine()
            
        except Exception as e:
            logger.error(f"Failed to initialize evaluator: {e}")
            raise
    
    def start_session(
        self,
        task_title: str,
        task_description: str,
        task_details: str,
        max_attempts: int = 5,
    ) -> Dict[str, Any]:
        """
        Start a new interactive evaluation session.

        Args:
            task_title: Title of the task
            task_description: Description of what to learn
            task_details: Detailed explanation and context
            max_attempts: Maximum number of attempts allowed

        Returns:
            Dict with `session_id` and first `question`
        """
        # Build in-memory context from user input
        context = TaskEvaluationContext(
            task_title=task_title,
            task_description=task_description,
            task_details=task_details,
            key_concepts=self._extract_key_concepts(task_details),
        )

        # Create session
        session_id = str(uuid.uuid4())
        session = EvaluationSession(
            session_id=session_id,
            task_title=task_title,
            task_description=task_description,
            task_details=task_details,
            context=context,
            max_attempts=max_attempts,
            state=SessionState.ASKING,
        )

        # Generate first Socratic question
        question = self._generate_question_for_session(session)

        session.updated_at = datetime.utcnow()
        self.sessions[session_id] = session
        
        # Save to DB
        self.db.save_evaluation_session(session.model_dump(mode="json"))

        logger.info(f"Started session {session_id} for task: {task_title}")
        return {"session_id": session_id, "question": question}

    def handle_user_answer(self, session_id: str, user_answer: str) -> Dict[str, Any]:
        """
        Handle student answer for a given session.
        Always appends user_answer to history, even on fallback/error.

        Args:
            session_id: ID of the evaluation session
            user_answer: Student's text answer (the actual input from the user)

        Returns:
            Dict with evaluation result: next question, reward, or reschedule
        """
        session = self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return {"error": "session_not_found", "session_id": session_id}

        # Prevent processing if already complete
        if session.state == SessionState.COMPLETE:
            logger.info(f"Session {session_id} already complete")
            return {
                "session_id": session_id,
                "state": "complete",
                "message": "Evaluation session is complete",
            }

        # IMPORTANT: Always append the user's answer to history first
        # Even if analysis fails, we want to record that the user provided input
        session.answer_history.append(user_answer)
        session.attempts += 1
        session.state = SessionState.ANALYZING
        session.updated_at = datetime.utcnow()

        user_answer_preview = user_answer[:80] if user_answer else "(empty)"
        logger.debug(f"[{session_id}] Processing attempt {session.attempts}: '{user_answer_preview}'")

        # Analyze answer via Gemini
        analysis, mastery_score = self._analyze_answer_for_session(session, user_answer)
        session.mastery_score = mastery_score

        # Build result structure
        result: Dict[str, Any] = {
            "session_id": session_id,
            "mastery_score": mastery_score,
            "attempts": session.attempts,
            "feedback": analysis.answer_feedback,
        }

        # Generate specific feedback for missing concepts if score is low
        if mastery_score < 0.7:  # Below threshold for detailed feedback
            concept_feedback = self.scorer.generate_missing_concept_feedback(
                mastery_score=mastery_score,
                missing_concepts=analysis.missing_concepts,
                key_concepts=session.context.key_concepts,
                threshold=0.7
            )
            if concept_feedback:
                result["feedback"] = f"{analysis.answer_feedback} {concept_feedback}".strip()

        # Respect deterministic scorer decisions
        # First, check if max attempts reached
        if session.attempts >= session.max_attempts:
            if mastery_score < self.scorer.MASTERY_THRESHOLD:
                session.state = SessionState.COMPLETE
                logger.info(f"[{session_id}] Max attempts reached with score {mastery_score:.2f} - FAILED")
                result.update({
                    "state": "FAILED",
                    "message": f"Max attempts ({session.attempts}) reached. Score {mastery_score:.2f} below threshold.",
                    "weak_concepts": analysis.missing_concepts,
                    "misconceptions": analysis.misconceptions,
                })
                self.sessions[session_id] = session
                self.db.save_evaluation_session(session.model_dump(mode="json"))
                return result

        # Determine state by scorer
        state = self.scorer.determine_state(
            mastery_score=mastery_score,
            guessing_detected=analysis.guessing_detected,
            generic_answer=(analysis.logical_coherence < 0.5),
        )

        if state == "MASTERY_CONFIRMED":
            session.state = SessionState.COMPLETE
            logger.info(f"[{session_id}] MASTERY CONFIRMED with score {mastery_score:.2f}")
            result.update({
                "state": "MASTERY_CONFIRMED",
                "message": f"Congratulations! You've demonstrated mastery with a score of {mastery_score:.2f}",
                "concepts_covered": session.context.key_concepts,
            })

        elif state == "FAILED":
            session.state = SessionState.COMPLETE
            logger.info(f"[{session_id}] FAILED - score {mastery_score:.2f} below threshold")
            result.update({
                "state": "FAILED",
                "message": f"Evaluation unsuccessful. Score {mastery_score:.2f} below threshold. Please review and try again.",
                "weak_concepts": analysis.missing_concepts,
                "misconceptions": analysis.misconceptions,
            })

        else:  # CONTINUE
            session.state = SessionState.ASKING
            # Escalate depth level for deeper questioning
            if session.depth_level == "what":
                session.depth_level = "why"
                logger.debug(f"[{session_id}] Escalating to WHY level")
            elif session.depth_level == "why":
                session.depth_level = "how"
                logger.debug(f"[{session_id}] Escalating to HOW level")

            next_question = self._generate_question_for_session(session)
            logger.info(f"[{session_id}] CONTINUE - score {mastery_score:.2f}, attempt {session.attempts}/{session.max_attempts}")
            result.update({
                "state": "CONTINUE",
                "next_question": next_question,
                "message": f"Good effort! Let's explore deeper. Score: {mastery_score:.2f}",
            })

        session.updated_at = datetime.utcnow()
        self.sessions[session_id] = session
        self.db.save_evaluation_session(session.model_dump(mode="json"))
        return result

    # ------------------------------------------------------------------
    # Private Helper Methods
    # ------------------------------------------------------------------

    def _extract_key_concepts(self, task_details: str) -> list[str]:
        """Extract meaningful scientific/domain-specific key concepts from task details."""
        if not task_details or not task_details.strip():
            return []

        # Use the improved keyword extraction from prompts.py
        from src.evaluator.prompts import extract_keywords_from_text, clean_concepts

        # Extract keywords from task details
        extracted_keywords = extract_keywords_from_text(task_details)

        # Clean and filter the concepts
        cleaned_concepts = clean_concepts(extracted_keywords)

        # Additional domain-specific filtering
        domain_specific = []
        for concept in cleaned_concepts:
            concept_lower = concept.lower()
            # Prefer concepts that appear to be domain-specific (not too generic)
            if (len(concept) >= 5 and
                not concept_lower.startswith(('how ', 'what ', 'why ', 'when ', 'where ')) and
                concept_lower not in {'process', 'system', 'method', 'approach', 'technique'}):
                domain_specific.append(concept)

        # Return top 8 most relevant concepts
        return domain_specific[:8]

    def _generate_question_for_session(self, session: EvaluationSession) -> str:
        """
        Generate Socratic question for given session.
        
        Strategy for minimizing API usage:
        - First question (attempts == 0): Call Gemini API
        - Follow-up questions (attempts > 0): Use local templates
        - If score < threshold, target missing concepts in questions
        """
        
        # First question: use Gemini API with template fallback
        if session.attempts == 0:
            if self.llm is not None:
                logger.info(f"[{session.session_id}] Generating FIRST question via Gemini API (with template fallback)")
                prompt = build_question_prompt(
                    task_title=session.task_title,
                    task_description=session.task_description,
                    task_details=session.task_details,
                    key_concepts=session.context.key_concepts,
                    depth_level=session.depth_level,
                )
                question = self.llm.generate_question(
                    prompt=prompt,
                    max_tokens=200,
                    depth_level=session.depth_level,
                    key_concepts=session.context.key_concepts,
                    task_title=session.task_title,
                    task_details=session.task_details,
                    attempt_number=1
                )
                question = validate_question(question)
                if has_generic_question_terms(question) or not question_contains_concept(question, session.context.key_concepts):
                    logger.warning(f"[{session.session_id}] First question failed validation, regenerating locally.")
                    question = generate_template_question(
                        depth_level=session.depth_level,
                        key_concepts=session.context.key_concepts,
                        task_title=session.task_title,
                        task_details=session.task_details,
                        attempt_number=1
                    )
                    logger.debug(f"[{session.session_id}] Regenerated first question accepted: {question}")
            else:
                # No LLM available - use template directly
                logger.info(f"[{session.session_id}] Generating FIRST question via template (no LLM)")
                question = generate_template_question(
                    depth_level=session.depth_level,
                    key_concepts=session.context.key_concepts,
                    task_title=session.task_title,
                    task_details=session.task_details,
                    attempt_number=1
                )
        
        # Follow-up questions: use local templates (no API call)
        else:
            logger.info(f"[{session.session_id}] Generating follow-up question locally (attempt {session.attempts})")
            
            # Check if we have missing concepts from previous analysis
            missing_concepts = []
            if session.analysis_history:
                last_analysis = session.analysis_history[-1]
                missing_concepts = getattr(last_analysis, 'missing_concepts', [])
            
            # Prioritize questions about missing concepts if score was low
            question_concepts = session.context.key_concepts
            if missing_concepts and session.mastery_score and session.mastery_score < 0.7:
                # Focus on missing concepts for targeted remediation
                cleaned_missing = clean_concepts(missing_concepts)
                if cleaned_missing:
                    question_concepts = cleaned_missing + session.context.key_concepts
                    logger.info(f"[{session.session_id}] Targeting missing concepts: {cleaned_missing}")
            
            question = generate_followup_question(
                depth_level=session.depth_level,
                key_concepts=question_concepts,
                student_answer=session.answer_history[-1] if session.answer_history else "",
                attempt_number=session.attempts
            )

        session.question_history.append(question)
        logger.debug(
            f"[{session.session_id}] Generated {session.depth_level.upper()} question: "
            f"{question[:80]}..."
        )
        return question

    def _analyze_answer_for_session(
        self, session: EvaluationSession, student_answer: str
    ) -> tuple[LLMAnalysisResponse, float]:
        """
        Analyze student answer and compute mastery score.
        
        KEY: Ensures student_answer is passed to LLM for evaluation.
        Uses plain text feedback from Gemini with default scores.
        Never returns FAILED just due to model failure - returns 0.5 to continue session.

        Args:
            session: The evaluation session
            student_answer: The student's actual text response (USED FOR ANALYSIS)

        Returns:
            Tuple of (analysis_response, mastery_score)
            mastery_score is always a float 0.0-1.0 that allows continuation
        """
        # Get the question the student was answering
        if not session.question_history:
            logger.error(f"[{session.session_id}] No question in history")
            # Return default analysis with neutral score
            analysis = LLMAnalysisResponse(
                concept_coverage=0.5,
                logical_coherence=0.5,
                causal_reasoning=0.5,
                error_awareness=0.5,
                answer_feedback="Unable to evaluate answer - no previous question was asked.",
                guessing_detected=False,
                missing_concepts=session.context.key_concepts,
                misconceptions=[],
            )
            return analysis, 0.5

        student_question = session.question_history[-1]

        # CRITICAL: Build analysis prompt with the ACTUAL student answer
        # This is where the student's input gets passed to the LLM
        prompt = build_analysis_prompt(
            task_title=session.task_title,
            task_description=session.task_description,
            task_details=session.task_details,
            key_concepts=session.context.key_concepts,
            student_answer=student_answer,
            previous_answers=session.answer_history,
        )

        logger.debug(
            f"[{session.session_id}] Analyzing answer: '{student_answer[:60]}...' "
            f"for question: '{student_question[:60]}...'"
        )

        if self.llm is None:
            # No LLM available - use concept coverage only
            logger.info(f"[{session.session_id}] Analyzing answer using concept coverage only (no LLM)")
            local_concept_score = concept_coverage(student_answer, session.context.key_concepts)
            last_valid_score = getattr(session, 'mastery_score', 0.5) or 0.5
            
            fallback_analysis = LLMAnalysisResponse(
                concept_coverage=local_concept_score,
                logical_coherence=0.5,
                causal_reasoning=0.5,
                error_awareness=0.5,
                answer_feedback=f"Based on your answer, you covered {int(local_concept_score * 100)}% of key concepts.",
                guessing_detected=False,
                missing_concepts=[
                    concept for concept in session.context.key_concepts
                    if concept.lower() not in student_answer.lower()
                ],
                misconceptions=[],
            )
            
            mastery_score = self.scorer.compute_mastery_score(
                fallback_analysis,
                concept_score=local_concept_score,
                last_valid_score=last_valid_score
            )
            return fallback_analysis, mastery_score
        
        try:
            # Generate plain text analysis from Gemini (reduced tokens to minimize API usage)
            feedback_text = self.llm.generate(prompt, max_tokens=150)

            # Parse structured response from feedback text
            parsed_response = parse_analysis_response(feedback_text)

            # Extract and clean missing concepts from parsed response
            raw_missing = parsed_response.get("missing_concepts", [])
            cleaned_missing = clean_concepts(raw_missing) if raw_missing else []

            # If LLM didn't provide missing concepts, infer from concept coverage
            if not cleaned_missing:
                answer_lower = student_answer.lower()
                cleaned_missing = [
                    concept for concept in session.context.key_concepts
                    if concept.lower() not in answer_lower and clean_concepts([concept])
                ]

            # Limit to most relevant missing concepts (max 5)
            final_missing_concepts = cleaned_missing[:5]

            # Compute local concept coverage from student answer
            local_concept_score = concept_coverage(student_answer, session.context.key_concepts)
            logger.debug(f"[{session.session_id}] Local concept coverage: {local_concept_score:.3f}")

            # Create analysis response with parsed data
            analysis = LLMAnalysisResponse(
                concept_coverage=local_concept_score,
                logical_coherence=0.5,  # Default: neutral
                causal_reasoning=0.5,  # Default: neutral
                error_awareness=0.5,  # Default: neutral
                answer_feedback=(parsed_response.get("strengths", "") + " " + parsed_response.get("weaknesses", "")).strip(),
                guessing_detected=False,  # Default: not detected
                missing_concepts=final_missing_concepts,
                misconceptions=[],  # Will be determined by scorer
            )

            session.analysis_history.append(analysis)

            logger.debug(
                f"[{session.session_id}] Analysis received: feedback='{feedback_text[:80]}...', "
                f"parsed_score={parsed_response.get('score')}, missing_concepts={final_missing_concepts}"
            )

            # Get last valid score for fallback (use session mastery_score if available, else 0.5)
            last_valid_score = getattr(session, 'mastery_score', 0.5) or 0.5

            # Compute hybrid mastery score with robust fallback
            mastery_score = self.scorer.compute_mastery_score(
                analysis,
                concept_score=local_concept_score,
                last_valid_score=last_valid_score
            )
            logger.info(
                f"[{session.session_id}] Mastery score: {mastery_score:.3f} ({int(mastery_score * 100)}%) | "
                f"LLM: {parsed_response.get('score', 'N/A')} | "
                f"Concepts: {local_concept_score:.3f} | "
                f"Covered: {[c for c in session.context.key_concepts if c.lower() in student_answer.lower()]} | "
                f"Missing: {final_missing_concepts}"
            )
            return analysis, mastery_score
        except Exception as e:
            logger.warning(
                f"[{session.session_id}] Analysis pipeline error ({type(e).__name__}): {e}. "
                f"Using robust fallback scoring to continue session."
            )

            # Compute local concept coverage even on error
            local_concept_score = concept_coverage(student_answer, session.context.key_concepts)

            # Get last valid score for fallback
            last_valid_score = getattr(session, 'mastery_score', 0.5) or 0.5

            # Create fallback analysis with local concept analysis
            fallback_analysis = LLMAnalysisResponse(
                concept_coverage=local_concept_score,  # Use actual concept coverage
                logical_coherence=0.5,
                causal_reasoning=0.5,
                error_awareness=0.5,
                answer_feedback=f"Analysis temporarily unavailable. Based on your answer, you covered {int(local_concept_score * 100)}% of key concepts. Continuing evaluation.",
                guessing_detected=False,
                missing_concepts=[
                    concept for concept in session.context.key_concepts
                    if concept.lower() not in student_answer.lower()
                ],
                misconceptions=[],
            )

            session.analysis_history.append(fallback_analysis)

            # Use robust fallback scoring: 0.4 * concept_coverage + 0.6 * last_valid_score
            mastery_score = self.scorer.compute_mastery_score(
                fallback_analysis,
                concept_score=local_concept_score,
                last_valid_score=last_valid_score
            )

            logger.info(f"[{session.session_id}] Using fallback score {mastery_score:.3f} ({int(mastery_score * 100)}%) to continue session")
            return fallback_analysis, mastery_score

    def get_session(self, session_id: str) -> Optional[EvaluationSession]:
        """Get session by ID."""
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Try to load from DB
        session_data = self.db.get_evaluation_session(session_id)
        if session_data:
            # Remove _id from mongo doc
            if "_id" in session_data:
                del session_data["_id"]
            session = EvaluationSession(**session_data)
            self.sessions[session_id] = session
            return session
            
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True
        return False

    def evaluate(
        self,
        session_duration_minutes: int,
        focus_score: float,
        completed_tasks: int,
        skipped_tasks: int,
    ) -> EvaluationResult:
        """
        Evaluate a completed study session based on metrics.
        
        Args:
            session_duration_minutes: Duration of the study session
            focus_score: Average focus score (0-100)
            completed_tasks: Number of tasks completed
            skipped_tasks: Number of tasks skipped
            
        Returns:
            EvaluationResult with level and score
        """
        total_tasks = completed_tasks + skipped_tasks
        completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        focus_normalized = focus_score / 100.0
        
        duration_score = min(session_duration_minutes / 60.0, 1.0)
        
        weighted_score = (
            0.4 * completion_rate +
            0.35 * focus_normalized +
            0.25 * duration_score
        )
        
        score_100 = round(weighted_score * 100)
        
        if score_100 >= 85:
            level = "excellent"
        elif score_100 >= 70:
            level = "good"
        elif score_100 >= 50:
            level = "fair"
        else:
            level = "needs_improvement"
        
        feedback_parts = []
        if completion_rate >= 0.8:
            feedback_parts.append("Great job completing most of your tasks!")
        elif completion_rate >= 0.5:
            feedback_parts.append("You completed some tasks, but consider reducing the scope.")
        else:
            feedback_parts.append("Low task completion rate. Try breaking tasks into smaller pieces.")
            
        if focus_normalized >= 0.75:
            feedback_parts.append("Excellent focus throughout the session.")
        elif focus_normalized >= 0.5:
            feedback_parts.append("Focus was moderate. Try minimizing distractions.")
        else:
            feedback_parts.append("Focus was low. Consider shorter sessions with breaks.")
        
        feedback = " ".join(feedback_parts)
        
        state = EvaluationState.MASTERY_CONFIRMED if score_100 >= 70 else EvaluationState.FAILED
        
        reward = None
        if state == EvaluationState.MASTERY_CONFIRMED:
            xp = int(score_100 * 1.5)
            reward = RewardPayload(
                learning_points=xp,
                streak_increment=1,
                concepts_covered=[]
            )
        
        reschedule = None
        if state == EvaluationState.FAILED:
            reschedule = ReschedulePayload(
                action="REVIEW",
                reason="Session score below threshold. Review recommended.",
                weak_concepts=["focus", "task_completion"],
                misconceptions=[]
            )
        
        return EvaluationResult(
            state=state,
            mastery_score=weighted_score,
            questions_asked=0,
            feedback=feedback,
            next_question=None,
            reward=reward,
            reschedule=reschedule
        )
