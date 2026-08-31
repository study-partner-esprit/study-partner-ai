"""EvaluatorAgent unit tests — no LLM, no broker.

The agent is exercised with ``require_llm=False`` (local concept-coverage
scoring) so the multi-turn Socratic state machine can be pinned deterministically.
"""

from __future__ import annotations

from agents.evaluator.agent import EvaluatorAgent
from agents.evaluator.schemas import SessionState


def make_agent(max_attempts: int = 5) -> EvaluatorAgent:
    return EvaluatorAgent(require_llm=False)


def test_start_session_returns_question_and_session():
    agent = make_agent()
    result = agent.start_session(
        task_title="Photosynthesis",
        task_description="How plants convert sunlight to energy",
        task_details=(
            "Photosynthesis occurs in chloroplasts. Light reactions produce "
            "ATP and NADPH. Calvin cycle fixes CO2 into glucose."
        ),
    )
    assert result["session_id"]
    assert result["question"]
    session = agent.get_session(result["session_id"])
    assert session is not None
    assert session.task_title == "Photosynthesis"
    assert session.context.key_concepts
    assert session.state == SessionState.ASKING


def test_handle_user_answer_returns_state_and_score():
    agent = make_agent()
    sid = agent.start_session(
        task_title="Python Lists",
        task_description="How Python lists work",
        task_details=(
            "Lists are ordered mutable sequences written with square brackets; "
            "they support indexing, slicing, append, extend, insert, remove, pop."
        ),
        max_attempts=5,
    )["session_id"]

    result = agent.handle_user_answer(
        sid,
        "Python lists are ordered collections that can hold mixed types. "
        "You create them with square brackets. Methods include append, insert, "
        "remove, and pop. Lists support indexing and slicing.",
    )
    assert result["state"] in ("MASTERY_CONFIRMED", "CONTINUE", "FAILED")
    assert 0.0 <= result["mastery_score"] <= 1.0
    assert result["session_id"] == sid
    assert len(agent.get_session(sid).answer_history) >= 1


def test_answer_appended_to_history_before_final_state():
    agent = make_agent()
    sid = agent.start_session(
        task_title="History",
        task_description="World War II",
        task_details="World War II started in 1939 and ended in 1945, involving many countries.",
        max_attempts=5,
    )["session_id"]

    r1 = agent.handle_user_answer(sid, "WW2 was a global war from 1939 to 1945.")
    s1 = agent.get_session(sid)
    assert "1939" in s1.answer_history[0]
    assert len(s1.answer_history) == 1
    if r1["state"] == "CONTINUE":
        agent.handle_user_answer(sid, "Major powers included the Allies and Axis.")
        assert len(agent.get_session(sid).answer_history) == 2


def test_max_attempts_enforced():
    agent = make_agent(max_attempts=2)
    sid = agent.start_session(
        task_title="Test",
        task_description="Desc",
        task_details="very short details for max attempt testing.",
        max_attempts=2,
    )["session_id"]

    agent.handle_user_answer(sid, "no")
    r2 = agent.handle_user_answer(sid, "no")
    assert r2["state"] in ("FAILED", "complete")


def test_session_not_found():
    agent = make_agent()
    result = agent.handle_user_answer("nonexistent-session-id", "test answer")
    assert result.get("error") == "session_not_found"


def test_already_complete_session_rejected():
    agent = make_agent()
    sid = agent.start_session(
        task_title="Test",
        task_description="Desc",
        task_details="some content here for testing purposes.",
        max_attempts=3,
    )["session_id"]
    agent.handle_user_answer(sid, "no")
    session = agent.get_session(sid)
    session.state = SessionState.COMPLETE
    agent.sessions[sid] = session
    result = agent.handle_user_answer(sid, "should be rejected")
    assert result["state"] == "complete"


def test_delete_session_removes_from_memory():
    agent = make_agent()
    sid = agent.start_session(
        task_title="Test",
        task_description="Desc",
        task_details="delete test session no save needed.",
        max_attempts=3,
    )["session_id"]
    assert sid in agent.sessions
    assert agent.delete_session(sid) is True
    assert sid not in agent.sessions


def test_multiple_sessions_run_independently():
    agent = make_agent()
    r1 = agent.start_session(task_title="Topic A", task_description="Desc A",
                             task_details="content for topic A here.", max_attempts=5)
    r2 = agent.start_session(task_title="Topic B", task_description="Desc B",
                             task_details="content for topic B here.", max_attempts=5)
    assert r1["session_id"] != r2["session_id"]
    agent.handle_user_answer(r1["session_id"], "answer for A")
    agent.handle_user_answer(r2["session_id"], "answer for B")
    assert agent.get_session(r1["session_id"]).answer_history == ["answer for A"]
    assert agent.get_session(r2["session_id"]).answer_history == ["answer for B"]


def test_concept_extraction_from_task_details():
    agent = make_agent()
    concepts = agent._extract_key_concepts(
        "Machine learning uses neural networks, gradient descent, and "
        "backpropagation to train models on labeled data."
    )
    assert len(concepts) > 0
