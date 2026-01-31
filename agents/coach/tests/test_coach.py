"""Tests for the coach agent."""
import pytest
from agents.coach.coach import CoachAgent


class TestCoachAgent:
    """Test cases for CoachAgent."""

    def test_coach_initialization(self):
        """Test coach agent can be initialized."""
        agent = CoachAgent()
        assert agent is not None
        assert agent.agent_type == "coach"

    @pytest.mark.asyncio
    async def test_provide_guidance(self):
        """Test coach can provide guidance."""
        agent = CoachAgent()
        decision = await agent.provide_guidance(
            user_id="test_user",
            current_task={"title": "Learn variables"},
            context={"focus_level": 0.8}
        )
        
        assert decision is not None
        assert decision.agent_type == "coach"
        assert decision.decision_type == "coach"
        assert "message" in decision.content

    @pytest.mark.asyncio
    async def test_motivational_support(self):
        """Test coach can provide motivation."""
        agent = CoachAgent()
        decision = await agent.provide_motivation(
            user_id="test_user",
            motivation_level=0.4,
            progress=0.6
        )
        
        assert decision is not None
        assert decision.confidence > 0

    @pytest.mark.asyncio
    async def test_fatigue_detection(self):
        """Test coach can detect fatigue."""
        agent = CoachAgent()
        decision = await agent.detect_fatigue(
            user_id="test_user",
            session_duration=90,
            activity_metrics={"clicks": 10, "focus_time": 80}
        )
        
        assert decision is not None
        assert "fatigue_detected" in decision.content
