"""Tests for the planner agent."""
import pytest
from agents.planner.planner import PlannerAgent


class TestPlannerAgent:
    """Test cases for PlannerAgent."""

    def test_planner_initialization(self):
        """Test planner agent can be initialized."""
        agent = PlannerAgent()
        assert agent is not None
        assert agent.agent_type == "planner"

    @pytest.mark.asyncio
    async def test_create_study_plan(self):
        """Test planner can create a study plan."""
        agent = PlannerAgent()
        decision = await agent.create_plan(
            user_id="test_user",
            goals=["Learn Python"],
            available_time=60,
            difficulty_level="beginner"
        )
        
        assert decision is not None
        assert decision.agent_type == "planner"
        assert decision.decision_type == "plan"
        assert decision.confidence > 0

    @pytest.mark.asyncio
    async def test_adapt_plan_to_progress(self):
        """Test planner can adapt plan based on progress."""
        agent = PlannerAgent()
        decision = await agent.adapt_plan(
            current_plan={"tasks": []},
            progress_data={"completed": 3, "total": 5}
        )
        
        assert decision is not None
        assert decision.agent_type == "planner"
