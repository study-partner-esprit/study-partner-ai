"""PlannerRequest schema tests (F02 / PLAN-02, BLOOM-10)."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from workers.schemas import (
    AVAILABLE_MINUTES_MAX,
    CONCEPTS_MAX_ITEMS,
    CONCEPT_MAX_CHARS,
    GOAL_MAX_CHARS,
    LEVEL_MAX_CHARS,
    PlannerRequest,
    WEAK_COMPETENCIES_MAX_ITEMS,
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
    assert pi.weak_competencies == []


def _wc(**kw):
    base = {
        "topic_id": "sorting",
        "topic_title": "Sorting Algorithms",
        "knowledge_type": "conceptual",
        "scores": {"remember": 0.9, "understand": 0.8, "apply": 0.4},
        "current_level": "apply",
        "unlocked_levels": ["remember", "understand", "apply"],
    }
    base.update(kw)
    return base


def test_weak_competencies_default_empty():
    r = PlannerRequest(goal="g")
    assert r.weak_competencies == []


def test_weak_competencies_bounded_by_count():
    ok = PlannerRequest(
        goal="g",
        weak_competencies=[_wc(topic_id=f"t{i}") for i in range(WEAK_COMPETENCIES_MAX_ITEMS)],
    )
    assert len(ok.weak_competencies) == WEAK_COMPETENCIES_MAX_ITEMS
    with pytest.raises(ValidationError):
        PlannerRequest(
            goal="g",
            weak_competencies=[
                _wc(topic_id=f"t{i}") for i in range(WEAK_COMPETENCIES_MAX_ITEMS + 1)
            ],
        )


def test_weak_competencies_length_bounds():
    with pytest.raises(ValidationError):
        PlannerRequest(goal="g", weak_competencies=[_wc(topic_id="x" * (LEVEL_MAX_CHARS + 1))])


def test_weak_competencies_scores_must_be_in_range():
    with pytest.raises(ValidationError):
        PlannerRequest(goal="g", weak_competencies=[_wc(scores={"remember": 1.5})])
    with pytest.raises(ValidationError):
        PlannerRequest(goal="g", weak_competencies=[_wc(scores={"remember": -0.1})])
    PlannerRequest(goal="g", weak_competencies=[_wc(scores={"remember": 0.0, "apply": 1.0})])


def test_weak_competencies_forbidden_extra_fields():
    with pytest.raises(ValidationError):
        PlannerRequest(goal="g", weak_competencies=[_wc(injected="tag")])


def test_to_planner_input_maps_weak_competencies():
    from agents.planner.models.task_graph import WeakCompetency

    r = PlannerRequest(goal="g", weak_competencies=[_wc()])
    pi = r.to_planner_input(user_id="u-9")
    assert len(pi.weak_competencies) == 1
    wc = pi.weak_competencies[0]
    assert isinstance(wc, WeakCompetency)
    assert wc.topic_id == "sorting"
    assert wc.topic_title == "Sorting Algorithms"
    assert wc.knowledge_type == "conceptual"
    assert wc.current_level == "apply"
    # unlocked_levels is coerced to a tuple for the model
    assert tuple(wc.unlocked_levels) == ("remember", "understand", "apply")
    assert wc.scores["apply"] == 0.4
