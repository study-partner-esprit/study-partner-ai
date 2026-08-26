"""Planner output validation pipeline (F02 / PLAN-04 + PLAN-05).

Structural schema validation happens in Pydantic (TaskGraph/AtomicTask).
This module adds semantic checks that the schema cannot express, and is the
single gate used by BOTH the LLM path (with one correction retry) and the
deterministic fallback path.
"""

from __future__ import annotations

from typing import List

from agents.planner.models.task_graph import PlannerOutput


class PlanValidationError(ValueError):
    """Raised when a planner output fails semantic validation."""

    def __init__(self, problems: List[str]):
        self.problems = problems
        super().__init__("; ".join(problems) or "plan validation failed")


def validate_plan_output(
    output: PlannerOutput,
    *,
    available_minutes: int | None = None,
    max_total_multiplier: float = 1.5,
) -> List[str]:
    """Return a list of semantic problems (empty list = valid)."""
    problems: List[str] = []
    tasks = output.task_graph.tasks if output.task_graph else []

    if not tasks:
        problems.append("plan contains no tasks")

    titles: set[str] = set()
    ids: set[str] = set()
    for i, task in enumerate(tasks):
        if not task.title.strip():
            problems.append(f"task[{i}] has empty title")
        if task.title.lower() in titles:
            problems.append(f"duplicate task title: {task.title!r}")
        titles.add(task.title.lower())

        if task.id in ids:
            problems.append(f"duplicate task id: {task.id!r}")
        ids.add(task.id)

        # AtomicTask bounds (5–45 min) are enforced by Pydantic; belt & braces:
        if not 5 <= task.estimated_minutes <= 45:
            problems.append(f"task[{i}] duration out of range: {task.estimated_minutes}")

    # Prerequisite graph coherence: every prerequisite must reference an
    # existing task id or title (checked after all ids are collected).
    known = ids | titles
    for i, task in enumerate(tasks):
        for prereq in task.prerequisites:
            if prereq and prereq not in known:
                problems.append(
                    f"task[{i}] ({task.title!r}) references unknown prerequisite {prereq!r}"
                )

    if available_minutes is not None and tasks:
        total = sum(t.estimated_minutes for t in tasks)
        if total > available_minutes * max_total_multiplier:
            problems.append(
                f"total estimated minutes {total} exceeds budget "
                f"{available_minutes} x{max_total_multiplier}"
            )

    if output.clarification_required and tasks:
        problems.append("clarification_required cannot be true when tasks are present")

    return problems
