"""
Reflection generation service using the shared LiteLLM client.
"""

import logging
import json
from datetime import datetime, timezone

from pymongo.errors import PyMongoError

from src.reflection.database import get_reflections_collection
from src.reflection.services.trend_service import compute_trends
from utils.llm_client import (
    LLMRequestError,
    MissingMockResponderError,
    agent_config,
    ask,
)

logger = logging.getLogger(__name__)

# Persona shown to the `reflection` model (litellm/config.yaml); the data
# itself is passed as the user prompt via build_prompt().
REFLECTION_SYSTEM_PROMPT = (
    "You are an intelligent study coach writing a personalized weekly "
    "reflection. Be specific, human, and encouraging. Reference the actual "
    "numbers in your analysis."
)


def build_prompt(user_id: str, trends: dict) -> str:
    """Build the LLM prompt for reflection generation."""
    t = trends["trends"]
    score = trends["progression_score"]
    weeks = trends["weeks_analyzed"]

    return f"""You are an intelligent study coach analyzing a student's weekly learning data.

Student ID: {user_id}
Weeks analyzed: {weeks}
Overall progression score: {score}/100

Performance data compared to last week:

Focus score (0 to 1, higher is better):
  - This week: {t['focus']['current']}
  - Last week: {t['focus']['previous']}
  - Change: {t['focus']['change_percent']:+.1f}%
  - Trend: {t['focus']['trend']}

Fatigue score (0 to 1, lower is better):
  - This week: {t['fatigue']['current']}
  - Last week: {t['fatigue']['previous']}
  - Change: {t['fatigue']['change_percent']:+.1f}%
  - Trend: {t['fatigue']['trend']}

XP earned:
  - This week: {t['xp']['current']}
  - Last week: {t['xp']['previous']}
  - Change: {t['xp']['change_percent']:+.1f}%
  - Trend: {t['xp']['trend']}

Total study minutes:
  - This week: {t['study_minutes']['current']}
  - Last week: {t['study_minutes']['previous']}
  - Change: {t['study_minutes']['change_percent']:+.1f}%
  - Trend: {t['study_minutes']['trend']}

Based on this data, generate a personalized and intelligent weekly reflection.
Be specific, human, and encouraging. Reference the actual numbers in your analysis.
Respond ONLY in this exact JSON format, no extra text, no markdown:

{{
  "summary": "2-3 sentences overall assessment referencing the actual data",
  "strengths": ["specific strength 1", "specific strength 2"],
  "weaknesses": ["specific weakness 1 or empty list if none"],
  "tips": ["concrete actionable tip 1", "concrete actionable tip 2", "concrete actionable tip 3"]
}}"""


def generate_reflection(user_id: str) -> dict:
    """
    Generate a personalized weekly reflection using Groq LLM.
    Stores the reflection in the database.
    """
    try:
        trends = compute_trends(user_id)
        logger.info(f"Trends computed for user {user_id}: {trends.get('status', 'ok')}")

        if trends.get("status") in ("error", "insufficient_data"):
            logger.warning(f"Trends error or insufficient data for {user_id}: {trends}")
            return trends

        # Build prompt and call the shared `reflection` LLM
        prompt = build_prompt(user_id, trends)
        logger.debug("Calling reflection LLM for reflection generation")

        raw = ""
        try:
            raw = ask("reflection", REFLECTION_SYSTEM_PROMPT, prompt).strip()
        except (LLMRequestError, MissingMockResponderError) as e:
            return {
                "status": "error",
                "detail": f"Reflection LLM unavailable: {str(e)}",
            }
        logger.debug(f"Reflection response received: {raw[:200]}...")

        # Extract JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode failed: {str(e)}")
        return {"status": "error", "detail": f"LLM returned invalid JSON: {str(e)}", "raw_response": raw}
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"Reflection error: {error_msg}")
        return {"status": "error", "detail": f"Reflection error: {str(e)}", "traceback": error_msg}

    # Build reflection document
    reflection = {
        "user_id": user_id,
        "period": "weekly",
        "progression_score": trends["progression_score"],
        "summary": parsed.get("summary", ""),
        "strengths": parsed.get("strengths", []),
        "weaknesses": parsed.get("weaknesses", []),
        "tips": parsed.get("tips", []),
        "trends_snapshot": trends["trends"],
        "generated_by": agent_config("reflection").model,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    # Store in database
    try:
        collection = get_reflections_collection()
        collection.insert_one(reflection)
        logger.info(f"Reflection stored for user {user_id}")
    except PyMongoError as e:
        logger.error(f"Failed to store reflection: {e}")
        reflection["storage_warning"] = str(e)

    return reflection


def get_user_reflections(user_id: str, limit: int = 10) -> list:
    """Retrieve stored reflections for a user."""
    try:
        collection = get_reflections_collection()
        return list(
            collection.find({"user_id": user_id})
            .sort("created_at", -1)
            .limit(limit)
        )
    except Exception as e:
        logger.error(f"Failed to get reflections: {e}")
        return []