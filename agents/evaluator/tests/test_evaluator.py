"""Tests for the evaluator agent."""
import pytest
from agents.evaluator.evaluator import EvaluatorAgent


class TestEvaluatorAgent:
    """Test cases for EvaluatorAgent."""

    def test_evaluator_initialization(self):
        """Test evaluator agent can be initialized."""
        agent = EvaluatorAgent()
        assert agent is not None
        assert agent.agent_type == "evaluator"

    @pytest.mark.asyncio
    async def test_assess_progress(self):
        """Test evaluator can assess student progress."""
        agent = EvaluatorAgent()
        decision = await agent.assess_progress(
            user_id="test_user",
            completed_tasks=[{"id": "1"}, {"id": "2"}],
            time_spent=90
        )
        
        assert decision is not None
        assert decision.agent_type == "evaluator"
        assert decision.decision_type == "evaluate"
        assert "completion_rate" in decision.content

    @pytest.mark.asyncio
    async def test_provide_feedback(self):
        """Test evaluator can provide constructive feedback."""
        agent = EvaluatorAgent()
        decision = await agent.provide_feedback(
            user_id="test_user",
            task_id="task_123",
            performance_data={"score": 0.85, "time": 30}
        )
        
        assert decision is not None
        assert "overall_rating" in decision.content

    @pytest.mark.asyncio
    async def test_concept_mastery_evaluation(self):
        """Test evaluator can assess concept mastery."""
        agent = EvaluatorAgent()
        decision = await agent.evaluate_concept_mastery(
            user_id="test_user",
            concept="Python Variables",
            quiz_results={"score": 0.8, "attempts": 2}
        )
        
        assert decision is not None
        assert "mastery_level" in decision.content
        assert decision.content["concept"] == "Python Variables"
