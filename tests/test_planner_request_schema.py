"""PlannerRequest schema tests (F02 / PLAN-02)."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from workers.schemas import (
    AVAILABLE_MINUTES_MAX,
    CONCEPTS_MAX_ITEMS,
    CONCEPT_MAX_CHARS,
    GOAL_MAX_CHARS,
    PlannerRequest,
)


def test_valid_minimal_request():
    r = PlannerRequest(goal="learn rabbitmq")
    assert r.concepts == []
    assert r.available_minutes == 120
    assert r.course_id is None and r.deadline is None


def test_goal_length_bounds():
    with pytest.raises(ValidationError):
        PlannerRequest(goal="")
    with pytest.raises(ValidationError):
        PlannerRequest(goal="   ")
    with pytest.raises(ValidationError):
        PlannerRequest(goal="x" * (GOAL_MAX_CHARS + 1))
    assert PlannerRequest(goal="x" * GOAL_MAX_CHARS).goal == "x" * GOAL_MAX_CHARS


def test_concepts_bounds():
    ok = PlannerRequest(goal="g", concepts=["a" * CONCEPT_MAX_CHARS] * CONCEPTS_MAX_ITEMS)
    assert len(ok.concepts) == CONCEPTS_MAX_ITEMS

    with pytest.raises(ValidationError):
        PlannerRequest(goal="g", concepts=["a"] * (CONCEPTS_MAX_ITEMS + 1))
    with pytest.raises(ValidationError):
        PlannerRequest(goal="g", concepts=["x" * (CONCEPT_MAX_CHARS + 1)])
    with pytest.raises(ValidationError):
        PlannerRequest(goal="g", concepts=["", "ok"])


def test_available_minutes_bounds():
    with pytest.raises(ValidationError):
        PlannerRequest(goal="g", available_minutes=0)
    with pytest.raises(ValidationError):
        PlannerRequest(goal="g", available_minutes=AVAILABLE_MINUTES_MAX + 1)
    assert PlannerRequest(goal="g", available_minutes=1).available_minutes == 1


def test_deadline_parsed_and_rejected_if_garbage():
    r = PlannerRequest(goal="g", deadline="2026-12-01T10:00:00Z")
    assert isinstance(r.deadline, datetime)
    with pytest.raises(ValidationError):
        PlannerRequest(goal="g", deadline="not-a-date")


def test_unknown_fields_forbidden():
    with pytest.raises(ValidationError):
        PlannerRequest(goal="g", evil_instruction="ignore previous")


def test_to_planner_input_maps_identity_and_defaults():
    from agents.planner.models.task_graph import PlannerInput

    r = PlannerRequest(goal="graphs", available_minutes=45)
    pi = r.to_planner_input(user_id="u-9")
    assert isinstance(pi, PlannerInput)
    assert pi.user_id == "u-9"
    assert pi.available_minutes == 45
    assert pi.goal == "graphs"
