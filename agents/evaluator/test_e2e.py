#!/usr/bin/env python3
"""
End-to-end tests for the EvaluatorAgent (Socratic + session evaluation).
All tests use require_llm=False to avoid external API dependencies.
"""

import sys
import logging

logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")

sys.path.insert(0, "src")

from evaluator.evaluator_agent import EvaluatorAgent
from evaluator.schemas import SessionState


def test_start_session():
    """Starting a session returns session_id and a first question."""
    agent = EvaluatorAgent(require_llm=False)
    result = agent.start_session(
        task_title="Photosynthesis",
        task_description="How plants convert sunlight to energy",
        task_details="Photosynthesis occurs in chloroplasts. Light reactions produce ATP and NADPH. Calvin cycle fixes CO2 into glucose.",
        max_attempts=5,
    )
    assert result["session_id"], "Expected session_id"
    assert result["question"], "Expected first question"
    assert len(result["question"]) > 10, "Question should be meaningful"

    sid = result["session_id"]
    session = agent.get_session(sid)
    assert session is not None
    assert session.task_title == "Photosynthesis"
    assert len(session.context.key_concepts) > 0
    assert session.state == SessionState.ASKING
    print(f"  PASS test_start_session (concepts: {session.context.key_concepts})")


def test_submit_answer():
    """Submitting an answer returns valid state + mastery_score."""
    agent = EvaluatorAgent(require_llm=False)
    result = agent.start_session(
        task_title="Python Lists",
        task_description="How Python lists work",
        task_details="Lists are ordered mutable sequences. items in square brackets. support indexing, slicing, append, extend, insert, remove, pop.",
        max_attempts=5,
    )
    sid = result["session_id"]

    r = agent.handle_user_answer(
        sid,
        "Python lists are ordered collections that can hold mixed types. "
        "You create them with square brackets. Methods include append to add "
        "at end, insert to add at position, remove to delete by value, and "
        "pop to remove by index. Lists support indexing and slicing.",
    )
    assert "state" in r, "Result should contain state"
    assert r["state"] in ("MASTERY_CONFIRMED", "CONTINUE", "FAILED"), f"Unexpected state: {r['state']}"
    assert 0.0 <= r["mastery_score"] <= 1.0, f"mastery_score out of range: {r['mastery_score']}"
    assert r["session_id"] == sid

    session = agent.get_session(sid)
    assert len(session.answer_history) >= 1
    assert len(session.question_history) >= 1
    print(f"  PASS test_submit_answer (state={r['state']}, score={r['mastery_score']:.3f})")


def test_answer_history_append():
    """Answers are appended to session history for tracking."""
    agent = EvaluatorAgent(require_llm=False)
    result = agent.start_session(
        task_title="History", task_description="World War II",
        task_details="World War II started in 1939 and ended in 1945 involving many countries.",
        max_attempts=5,
    )
    sid = result["session_id"]

    r1 = agent.handle_user_answer(sid, "WW2 was a global war from 1939 to 1945.")
    s1 = agent.get_session(sid)
    assert "1939" in s1.answer_history[0]
    assert len(s1.answer_history) == 1

    if r1["state"] == "CONTINUE":
        agent.handle_user_answer(sid, "Major powers included Allies and Axis.")
        s2 = agent.get_session(sid)
        assert len(s2.answer_history) == 2
    else:
        assert r1["state"] in ("MASTERY_CONFIRMED", "FAILED")

    print("  PASS test_answer_history_append")


def test_max_attempts_enforced():
    """Session stops after max_attempts even if all answers are poor."""
    agent = EvaluatorAgent(require_llm=False)
    result = agent.start_session(
        task_title="Test", task_description="Desc",
        task_details="very short details for max attempt testing.",
        max_attempts=2,
    )
    sid = result["session_id"]

    r1 = agent.handle_user_answer(sid, "no")
    r2 = agent.handle_user_answer(sid, "no")
    assert r2["state"] in ("FAILED", "complete"), f"Expected terminal state after max attempts, got {r2['state']}"
    print(f"  PASS test_max_attempts_enforced")


def test_session_not_found():
    """handle_user_answer returns error for unknown session."""
    agent = EvaluatorAgent(require_llm=False)
    r = agent.handle_user_answer("nonexistent-session-id", "test answer")
    assert r.get("error") == "session_not_found"
    print("  PASS test_session_not_found")


def test_already_complete_session():
    """handle_user_answer returns complete state for finished session."""
    agent = EvaluatorAgent(require_llm=False)
    result = agent.start_session(
        task_title="Test", task_description="Desc",
        task_details="some content here for testing purposes.",
        max_attempts=3,
    )
    sid = result["session_id"]

    agent.handle_user_answer(sid, "no")

    session = agent.get_session(sid)
    session.state = SessionState.COMPLETE
    agent.sessions[sid] = session

    r = agent.handle_user_answer(sid, "should be rejected")
    assert r["state"] == "complete", f"Expected 'complete', got {r['state']}"
    print("  PASS test_already_complete_session")


def test_evaluate_session():
    """Session evaluation uses duration, focus, completion metrics."""
    agent = EvaluatorAgent(require_llm=False)

    result = agent.evaluate(
        session_duration_minutes=45,
        focus_score=85,
        completed_tasks=8,
        skipped_tasks=2,
    )
    assert result.state in ("mastery_confirmed", "failed")
    assert 0.0 <= result.mastery_score <= 1.0
    assert result.feedback, "Expected feedback string"
    print(f"  PASS test_evaluate_session (state={result.state}, score={result.mastery_score:.2f})")


def test_evaluate_session_poor():
    """Poor session metrics yield failed state."""
    agent = EvaluatorAgent(require_llm=False)

    result = agent.evaluate(
        session_duration_minutes=5,
        focus_score=20,
        completed_tasks=0,
        skipped_tasks=10,
    )
    assert result.state == "failed"
    assert result.mastery_score < 0.5
    assert result.reschedule is not None
    print("  PASS test_evaluate_session_poor")


def test_evaluate_session_excellent():
    """Excellent session metrics yield mastery_confirmed."""
    agent = EvaluatorAgent(require_llm=False)

    result = agent.evaluate(
        session_duration_minutes=90,
        focus_score=95,
        completed_tasks=10,
        skipped_tasks=0,
    )
    assert result.state == "mastery_confirmed"
    assert result.mastery_score >= 0.7
    assert result.reward is not None
    assert result.reward.learning_points > 0
    print(f"  PASS test_evaluate_session_excellent (score={result.mastery_score:.2f}, xp={result.reward.learning_points})")


def test_delete_session():
    """Deleted sessions are removed from in-memory dict."""
    agent = EvaluatorAgent(require_llm=False)
    result = agent.start_session(
        task_title="Test", task_description="Desc",
        task_details="delete test session no save needed.",
        max_attempts=3,
    )
    sid = result["session_id"]

    assert sid in agent.sessions, "Session should be in memory"
    agent.delete_session(sid)
    assert sid not in agent.sessions, "Session should be removed from memory"
    print("  PASS test_delete_session")


def test_multiple_sessions():
    """Multiple independent sessions can run simultaneously."""
    agent = EvaluatorAgent(require_llm=False)

    r1 = agent.start_session(task_title="Topic A", task_description="Desc A",
                             task_details="content for topic A here.", max_attempts=5)
    r2 = agent.start_session(task_title="Topic B", task_description="Desc B",
                             task_details="content for topic B here.", max_attempts=5)

    assert r1["session_id"] != r2["session_id"]

    agent.handle_user_answer(r1["session_id"], "answer for A")
    agent.handle_user_answer(r2["session_id"], "answer for B")

    s1 = agent.get_session(r1["session_id"])
    s2 = agent.get_session(r2["session_id"])
    assert len(s1.answer_history) == 1
    assert len(s2.answer_history) == 1
    assert s1.answer_history[0] == "answer for A"
    assert s2.answer_history[0] == "answer for B"
    print("  PASS test_multiple_sessions")


def test_concept_extraction():
    """Key concepts are extracted from task details."""
    agent = EvaluatorAgent(require_llm=False)
    concepts = agent._extract_key_concepts(
        "Machine learning uses neural networks gradient descent "
        "and backpropagation to train models on labeled data."
    )
    assert len(concepts) > 0, "Expected at least one concept"
    print(f"  PASS test_concept_extraction (concepts: {concepts})")


if __name__ == "__main__":
    tests = [
        test_start_session,
        test_submit_answer,
        test_answer_history_append,
        test_max_attempts_enforced,
        test_session_not_found,
        test_already_complete_session,
        test_evaluate_session,
        test_evaluate_session_poor,
        test_evaluate_session_excellent,
        test_delete_session,
        test_multiple_sessions,
        test_concept_extraction,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(0 if failed == 0 else 1)
