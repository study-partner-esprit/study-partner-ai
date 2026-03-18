"""
ClarificationChecker — Detects ambiguous or incomplete planning requests.

Analyzes user goals and planning inputs to identify:
  1. Vague or overly broad goals
  2. Missing critical information (deadlines, time availability)
  3. Conflicting requirements
  4. Unrealistic expectations

Generates specific clarification questions to help the planner
produce a better study plan.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Words that indicate a vague goal
VAGUE_INDICATORS = [
    "stuff",
    "things",
    "something",
    "anything",
    "whatever",
    "etc",
    "and so on",
    "and more",
    "maybe",
    "kind of",
    "sort of",
    "a bit",
    "some",
    "a little",
    "general",
]

# Words that indicate a well-defined goal
SPECIFIC_INDICATORS = [
    "chapter",
    "module",
    "topic",
    "concept",
    "theorem",
    "algorithm",
    "function",
    "class",
    "method",
    "section",
    "exercise",
    "problem",
    "quiz",
    "exam",
    "test",
    "lecture",
    "lab",
    "project",
    "assignment",
    "homework",
]

# Minimum word count for a specific goal
MIN_GOAL_WORDS = 3
MIN_GOAL_CHARS = 10

# Maximum reasonable daily study hours
MAX_DAILY_HOURS = 12
MIN_DAILY_MINUTES = 15


class ClarificationResult:
    """Result of a clarification check."""

    def __init__(self):
        self.needs_clarification: bool = False
        self.questions: List[str] = []
        self.warnings: List[str] = []
        self.severity: str = "low"  # low, medium, high

    def add_question(self, question: str, severity: str = "medium"):
        self.needs_clarification = True
        self.questions.append(question)
        # Upgrade severity if needed
        severity_order = {"low": 0, "medium": 1, "high": 2}
        if severity_order.get(severity, 0) > severity_order.get(self.severity, 0):
            self.severity = severity

    def add_warning(self, warning: str):
        self.warnings.append(warning)

    def to_dict(self) -> Dict:
        return {
            "needs_clarification": self.needs_clarification,
            "questions": self.questions,
            "warnings": self.warnings,
            "severity": self.severity,
        }


class ClarificationChecker:
    """
    Checks planning inputs for ambiguity and generates clarification questions.
    """

    def check(
        self,
        goal: Optional[str] = None,
        deadline_iso: Optional[str] = None,
        available_minutes: Optional[int] = None,
        course_knowledge: Optional[Dict] = None,
    ) -> ClarificationResult:
        """
        Run all clarification checks on the planning input.

        Args:
            goal:              The user's study goal text.
            deadline_iso:      Deadline in ISO format.
            available_minutes: Total available study minutes.
            course_knowledge:  Structured course data (if from course upload).

        Returns:
            ClarificationResult with questions and warnings.
        """
        result = ClarificationResult()

        # If course_knowledge is provided, goal is less critical
        has_course = course_knowledge is not None and bool(course_knowledge)

        # Check goal clarity
        if goal:
            self._check_goal_vagueness(goal, result, has_course)
            self._check_goal_scope(goal, result)
        elif not has_course:
            result.add_question(
                "What specific topic or subject would you like to study? "
                "Please provide a clear learning goal.",
                severity="high",
            )

        # Check time feasibility
        if available_minutes is not None:
            self._check_time_feasibility(available_minutes, deadline_iso, result)

        # Check deadline
        if deadline_iso:
            self._check_deadline(deadline_iso, available_minutes, result)

        # Check course knowledge completeness
        if has_course:
            self._check_course_knowledge(course_knowledge, result)

        logger.info(
            "clarification_check_complete",
            extra={
                "needs_clarification": result.needs_clarification,
                "question_count": len(result.questions),
                "severity": result.severity,
            },
        )

        return result

    def check_goal(self, goal: str) -> bool:
        """
        Legacy API: Return True if goal is too vague and requires clarification.
        """
        result = self.check(goal=goal)
        return result.needs_clarification

    def check_plan_feasibility(self, tasks, available_minutes: int) -> bool:
        """
        Return True if the plan exceeds available time and needs negotiation.
        """
        total_minutes = sum(
            t.estimated_minutes if hasattr(t, "estimated_minutes") else 0 for t in tasks
        )
        return total_minutes > available_minutes

    # ------------------------------------------------------------------ #
    # Private check methods                                                #
    # ------------------------------------------------------------------ #

    def _check_goal_vagueness(
        self, goal: str, result: ClarificationResult, has_course: bool
    ) -> None:
        """Check if the goal is too vague or uses unclear language."""
        goal_lower = goal.lower().strip()
        words = goal_lower.split()

        # Too short
        if len(words) < MIN_GOAL_WORDS or len(goal.strip()) < MIN_GOAL_CHARS:
            if not has_course:
                result.add_question(
                    f"Your goal '{goal}' is quite brief. Could you be more specific? "
                    "For example: 'Learn Python data structures including lists, dictionaries, and sets'",
                    severity="high",
                )
            else:
                result.add_warning(
                    f"Goal '{goal}' is brief, but course materials will guide the plan."
                )
            return

        # Contains vague indicators
        found_vague = [word for word in VAGUE_INDICATORS if word in goal_lower]
        if found_vague:
            result.add_question(
                f"Your goal contains vague terms ({', '.join(found_vague)}). "
                "Could you specify exactly which topics or skills you want to focus on?",
                severity="medium",
            )

        # Check for specificity
        has_specific = any(word in goal_lower for word in SPECIFIC_INDICATORS)
        if not has_specific and len(words) < 8 and not has_course:
            result.add_warning(
                "Consider making your goal more specific by mentioning chapters, "
                "topics, or specific concepts you want to cover."
            )

    def _check_goal_scope(self, goal: str, result: ClarificationResult) -> None:
        """Check if the goal is too broad (trying to cover too much)."""
        goal_lower = goal.lower()

        # Check for "everything" or "all" patterns
        broad_patterns = [
            r"\beverything\b",
            r"\ball of\b",
            r"\bentire\b",
            r"\bwhole course\b",
            r"\bfrom scratch\b.*\bto advanced\b",
            r"\bbeginner to expert\b",
        ]

        for pattern in broad_patterns:
            if re.search(pattern, goal_lower):
                result.add_question(
                    "Your goal seems very broad, which might be hard to plan effectively. "
                    "Could you break it down into smaller milestones? "
                    "For example, focus on the first 3-4 chapters or specific topics.",
                    severity="medium",
                )
                break

    def _check_time_feasibility(
        self,
        available_minutes: int,
        deadline_iso: Optional[str],
        result: ClarificationResult,
    ) -> None:
        """Check if the available time is realistic."""
        if available_minutes < MIN_DAILY_MINUTES:
            result.add_question(
                f"You've specified only {available_minutes} minutes of study time. "
                "This might not be enough for meaningful learning. "
                "Could you allocate at least 30 minutes?",
                severity="medium",
            )

        if deadline_iso:
            try:
                from datetime import datetime, timezone

                deadline = datetime.fromisoformat(deadline_iso.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                days_until = (deadline - now).days

                if days_until > 0:
                    daily_minutes = available_minutes / days_until

                    if daily_minutes > MAX_DAILY_HOURS * 60:
                        result.add_question(
                            f"Your plan requires ~{int(daily_minutes)} minutes/day over {days_until} days. "
                            f"That's {daily_minutes / 60:.1f} hours/day, which is unsustainable. "
                            "Could you extend the deadline or reduce the scope?",
                            severity="high",
                        )
                    elif daily_minutes > 4 * 60:  # More than 4 hours/day
                        result.add_warning(
                            f"Heads up: your plan requires ~{daily_minutes / 60:.1f} hours/day. "
                            "Consider if this is sustainable for your schedule."
                        )
            except (ValueError, TypeError):
                pass  # Deadline validation handled elsewhere

    def _check_deadline(
        self,
        deadline_iso: str,
        available_minutes: Optional[int],
        result: ClarificationResult,
    ) -> None:
        """Check if the deadline is reasonable."""
        try:
            from datetime import datetime, timezone

            deadline = datetime.fromisoformat(deadline_iso.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            days_until = (deadline - now).days

            if days_until < 0:
                result.add_question(
                    "The deadline you specified has already passed. "
                    "Please provide a future deadline.",
                    severity="high",
                )
            elif days_until == 0:
                result.add_warning(
                    "Your deadline is today! The plan will be condensed. "
                    "Consider extending it if possible."
                )
            elif days_until < 2 and available_minutes and available_minutes > 120:
                result.add_warning(
                    f"You have only {days_until} day(s) until the deadline "
                    f"with {available_minutes} minutes of material. "
                    "The plan will prioritize the most important topics."
                )
        except (ValueError, TypeError):
            result.add_question(
                "The deadline format is invalid. Please provide a valid date.",
                severity="high",
            )

    def _check_course_knowledge(
        self, course_knowledge: Dict, result: ClarificationResult
    ) -> None:
        """Check if course knowledge data is complete enough for planning."""
        if not course_knowledge:
            return

        topics = course_knowledge.get("topics", [])
        if not topics:
            result.add_question(
                "The course materials didn't produce clear topics. "
                "Could you specify which topics you'd like to focus on?",
                severity="medium",
            )
            return

        # Check if there are too many topics without prioritization
        if len(topics) > 15:
            result.add_warning(
                f"The course has {len(topics)} topics. "
                "The planner will prioritize based on difficulty and dependencies, "
                "but you may want to specify which topics are most important to you."
            )
