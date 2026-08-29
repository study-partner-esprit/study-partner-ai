"""
LearningObjective model tests (F01 — Learning Objective Schema).

Node mirror: study-partner-api/study/src/tests/study.test.js
('learningObjective validator' block). Rules must stay identical on both
sides — see models/learning_objective.py's docstring.
"""

import pytest
from pydantic import ValidationError

from models.learning_objective import LearningObjective

VALID_PAYLOAD = {
    "objectiveId": "obj-1",
    "topicId": "topic-1",
    "knowledgeType": "conceptual",
    "bloomLevel": "apply",
    "verb": "Solve",
    "text": "Solve systems of linear equations using substitution.",
}


def test_accepts_a_valid_objective():
    obj = LearningObjective(**VALID_PAYLOAD)
    assert obj.objective_id == "obj-1"
    assert obj.bloom_level == "apply"
    assert obj.verb == "Solve"


def test_accepts_snake_case_payload_too():
    snake_payload = {
        "objective_id": "obj-2",
        "topic_id": "topic-1",
        "knowledge_type": "conceptual",
        "bloom_level": "apply",
        "verb": "Solve",
        "text": "Solve systems of linear equations using substitution.",
    }
    obj = LearningObjective(**snake_payload)
    assert obj.objective_id == "obj-2"


def test_rejects_verb_not_in_levels_verb_map():
    payload = {**VALID_PAYLOAD, "verb": "Design"}
    with pytest.raises(ValidationError, match="verb must be one of"):
        LearningObjective(**payload)


def test_rejects_bloom_level_outside_enum():
    payload = {**VALID_PAYLOAD, "bloomLevel": "memorize"}
    with pytest.raises(ValidationError, match="bloom_level must be one of"):
        LearningObjective(**payload)


def test_rejects_knowledge_type_outside_enum():
    payload = {**VALID_PAYLOAD, "knowledgeType": "emotional"}
    with pytest.raises(ValidationError, match="knowledge_type must be one of"):
        LearningObjective(**payload)


def test_rejects_non_measurable_phrasing():
    payload = {
        **VALID_PAYLOAD,
        "text": "Know how to solve systems of linear equations.",
    }
    with pytest.raises(ValidationError, match="non-measurable phrasing"):
        LearningObjective(**payload)


def test_rejects_empty_text():
    payload = {**VALID_PAYLOAD, "text": ""}
    with pytest.raises(ValidationError, match="non-empty string"):
        LearningObjective(**payload)


def test_rejects_text_over_200_chars():
    payload = {**VALID_PAYLOAD, "text": "Solve " + "x" * 200}
    with pytest.raises(ValidationError, match="exceeds 200 chars"):
        LearningObjective(**payload)


def test_rejects_verb_not_near_start_of_text():
    payload = {
        **VALID_PAYLOAD,
        "text": "Using substitution, systems of linear equations can be solved by students.",
    }
    with pytest.raises(ValidationError, match="must appear at/near the start"):
        LearningObjective(**payload)