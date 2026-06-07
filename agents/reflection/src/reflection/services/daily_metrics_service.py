"""
Daily metrics service for storing and retrieving daily study metrics.
"""

import logging
from src.reflection.database import get_daily_metrics_collection
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


def upsert_daily_metrics(data: dict) -> dict:
    """
    Upsert daily metrics for a user.
    
    Args:
        data: Dictionary containing user_id, date, total_study_minutes,
              avg_focus_score, avg_fatigue_score, xp_earned, sessions_count
    
    Returns:
        Dictionary with status and result details
    """
    filter_key = {
        "user_id": data["user_id"],
        "date": str(data["date"])
    }

    update_payload = {
        "$set": {
            "total_study_minutes": data["total_study_minutes"],
            "avg_focus_score": data["avg_focus_score"],
            "avg_fatigue_score": data["avg_fatigue_score"],
            "xp_earned": data["xp_earned"],
            "sessions_count": data["sessions_count"]
        }
    }

    try:
        collection = get_daily_metrics_collection()
        result = collection.update_one(
            filter_key,
            update_payload,
            upsert=True
        )
        if result.upserted_id:
            logger.info(f"Created daily metrics for user {data['user_id']} on {data['date']}")
            return {"status": "created", "id": str(result.upserted_id)}
        else:
            logger.info(f"Updated daily metrics for user {data['user_id']} on {data['date']}")
            return {"status": "updated", "matched": result.matched_count}

    except PyMongoError as e:
        logger.error(f"Failed to upsert daily metrics: {e}")
        return {"status": "error", "detail": str(e)}


def get_daily_metrics(user_id: str, date_str: str) -> dict | None:
    """Get daily metrics for a specific user and date."""
    try:
        collection = get_daily_metrics_collection()
        return collection.find_one({"user_id": user_id, "date": date_str})
    except PyMongoError as e:
        logger.error(f"Failed to get daily metrics: {e}")
        return None