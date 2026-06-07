"""
Aggregation service for weekly summaries and history.
"""

import logging
from datetime import date, timedelta
from typing import List, Optional

from src.reflection.database import get_daily_metrics_collection
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


def get_weekly_summary(user_id: str, reference_date: Optional[date] = None) -> dict:
    """
    Calculate weekly summary for the week containing reference_date.
    Defaults to current week.
    """
    if reference_date is None:
        reference_date = date.today()

    # Calculate Monday and Sunday of the week
    start_of_week = reference_date - timedelta(days=reference_date.weekday())
    end_of_week = start_of_week + timedelta(days=6)

    pipeline = [
        # Match user and date range
        {
            "$match": {
                "user_id": user_id,
                "date": {
                    "$gte": str(start_of_week),
                    "$lte": str(end_of_week)
                }
            }
        },
        # Group and calculate aggregates
        {
            "$group": {
                "_id": "$user_id",
                "total_study_minutes": {"$sum": "$total_study_minutes"},
                "total_xp_earned": {"$sum": "$xp_earned"},
                "total_sessions": {"$sum": "$sessions_count"},
                "avg_focus_score": {"$avg": "$avg_focus_score"},
                "avg_fatigue_score": {"$avg": "$avg_fatigue_score"},
                "days_studied": {"$sum": 1},
                "best_focus_day": {"$max": "$avg_focus_score"},
                "worst_fatigue_day": {"$max": "$avg_fatigue_score"},
            }
        },
        # Format response
        {
            "$project": {
                "_id": 0,
                "user_id": "$_id",
                "week_start": {"$literal": str(start_of_week)},
                "week_end": {"$literal": str(end_of_week)},
                "total_study_minutes": 1,
                "total_xp_earned": 1,
                "total_sessions": 1,
                "days_studied": 1,
                "avg_focus_score": {"$round": ["$avg_focus_score", 2]},
                "avg_fatigue_score": {"$round": ["$avg_fatigue_score", 2]},
                "best_focus_day": {"$round": ["$best_focus_day", 2]},
                "worst_fatigue_day": {"$round": ["$worst_fatigue_day", 2]},
            }
        }
    ]

    try:
        collection = get_daily_metrics_collection()
        results = list(collection.aggregate(pipeline))
        if not results:
            return {
                "user_id": user_id,
                "week_start": str(start_of_week),
                "week_end": str(end_of_week),
                "message": "No data found for this week"
            }
        return results[0]

    except PyMongoError as e:
        logger.error(f"Failed to get weekly summary: {e}")
        return {"status": "error", "detail": str(e)}


def get_all_weeks_summary(user_id: str) -> List[dict]:
    """
    Calculate aggregated summary per week for all user history.
    """
    pipeline = [
        {"$match": {"user_id": user_id}},
        # Convert date string to date object
        {
            "$addFields": {
                "date_obj": {"$dateFromString": {"dateString": "$date"}}
            }
        },
        # Group by ISO week
        {
            "$group": {
                "_id": {
                    "year": {"$isoWeekYear": "$date_obj"},
                    "week": {"$isoWeek": "$date_obj"}
                },
                "total_study_minutes": {"$sum": "$total_study_minutes"},
                "total_xp_earned": {"$sum": "$xp_earned"},
                "total_sessions": {"$sum": "$sessions_count"},
                "avg_focus_score": {"$avg": "$avg_focus_score"},
                "avg_fatigue_score": {"$avg": "$avg_fatigue_score"},
                "days_studied": {"$sum": 1},
            }
        },
        # Sort chronologically
        {"$sort": {"_id.year": 1, "_id.week": 1}},
        # Format output
        {
            "$project": {
                "_id": 0,
                "year": "$_id.year",
                "week": "$_id.week",
                "total_study_minutes": 1,
                "total_xp_earned": 1,
                "total_sessions": 1,
                "days_studied": 1,
                "avg_focus_score": {"$round": ["$avg_focus_score", 2]},
                "avg_fatigue_score": {"$round": ["$avg_fatigue_score", 2]},
            }
        }
    ]

    try:
        collection = get_daily_metrics_collection()
        return list(collection.aggregate(pipeline))

    except PyMongoError as e:
        logger.error(f"Failed to get all weeks summary: {e}")
        return [{"status": "error", "detail": str(e)}]