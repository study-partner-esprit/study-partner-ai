"""Repository for persisting and retrieving signal snapshots from MongoDB.

This module handles all database operations for user signals.
"""

from pymongo import MongoClient, DESCENDING
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import os

from services.signal_processing_service.signal_snapshot import SignalSnapshot
from utils.logger import get_logger

logger = get_logger(__name__)


class SignalRepository:
    """Handles persistence of SignalSnapshot data in MongoDB."""

    def __init__(self):
        """Initialize MongoDB connection."""
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("MONGO_DB_NAME", "study_partner")

        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["signals"]

        # Create index on user_id and timestamp for efficient queries
        self.collection.create_index([("user_id", 1), ("timestamp", DESCENDING)])

    def save_signal_snapshot(self, snapshot: SignalSnapshot) -> str:
        """
        Save a signal snapshot to MongoDB.

        Args:
            snapshot: The SignalSnapshot to persist

        Returns:
            The MongoDB document ID as a string
        """
        snapshot_dict = snapshot.model_dump()
        result = self.collection.insert_one(snapshot_dict)
        return str(result.inserted_id)

    def get_latest_signal_snapshot(self, user_id: str) -> Optional[SignalSnapshot]:
        """
        Retrieve the most recent signal snapshot for a user.

        Args:
            user_id: The user's unique identifier

        Returns:
            The latest SignalSnapshot, or None if no signals exist
        """
        # Use _id descending for more reliable ordering (ObjectIds are monotonically increasing)
        document = self.collection.find_one(
            {"user_id": user_id}, sort=[("_id", DESCENDING)]
        )

        if document is None:
            return None

        # Remove MongoDB's _id field before parsing
        document.pop("_id", None)

        return SignalSnapshot(**document)

    def get_signal_history(self, user_id: str, limit: int = 10) -> list[SignalSnapshot]:
        """
        Retrieve recent signal history for a user.

        Args:
            user_id: The user's unique identifier
            limit: Maximum number of snapshots to return

        Returns:
            List of SignalSnapshot objects, ordered by timestamp (newest first)
        """
        documents = self.collection.find(
            {"user_id": user_id}, sort=[("timestamp", DESCENDING)], limit=limit
        )

        snapshots = []
        for doc in documents:
            doc.pop("_id", None)
            snapshots.append(SignalSnapshot(**doc))

        return snapshots

    # ------------------------------------------------------------------ #
    # Productivity / trend analytics                                       #
    # ------------------------------------------------------------------ #

    def get_hourly_focus_profile(
        self,
        user_id: str,
        days_back: int = 30,
    ) -> Dict[int, float]:
        """
        Return a mapping of {hour_of_day: avg_focus_score} built from historical
        signal data for *user_id* over the last *days_back* days.

        Uses a MongoDB aggregation pipeline that:
          1. Filters to the relevant time window.
          2. Projects the hour-of-day from the timestamp.
          3. Groups and averages focus_score per hour.

        Args:
            user_id:   User to query.
            days_back: How many calendar days of history to include.

        Returns:
            Dict[int, float] mapping 0–23 → average focus score.
            Missing hours (no data) are omitted from the dict.
        """
        since = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "timestamp": {"$gte": since},
                }
            },
            {
                "$project": {
                    "hour": {"$hour": "$timestamp"},
                    "focus_score": 1,
                }
            },
            {
                "$group": {
                    "_id": "$hour",
                    "avg_focus": {"$avg": "$focus_score"},
                    "count": {"$sum": 1},
                }
            },
        ]
        try:
            results = list(self.collection.aggregate(pipeline))
            profile: Dict[int, float] = {
                int(r["_id"]): round(r["avg_focus"], 4)
                for r in results
                if r["_id"] is not None
            }
            logger.info(
                "hourly_focus_profile_built",
                extra={"user_id": user_id, "hours_with_data": len(profile)},
            )
            return profile
        except Exception as exc:
            logger.warning(
                "hourly_focus_profile_error",
                extra={"user_id": user_id, "error": str(exc)},
            )
            return {}

    def compute_focus_trend(
        self,
        user_id: str,
        window_minutes: int = 5,
    ) -> Optional[float]:
        """
        Compute the linear trend (slope) of focus_score over the last
        *window_minutes* minutes using ordinary least squares.

        A negative slope means focus is declining; positive means improving.

        Args:
            user_id:        User to compute the trend for.
            window_minutes: Time window to consider (default 5 minutes).

        Returns:
            Slope (float) or None if fewer than 2 data-points exist.
        """
        since = datetime.now(tz=timezone.utc) - timedelta(minutes=window_minutes)
        try:
            docs = list(
                self.collection.find(
                    {"user_id": user_id, "timestamp": {"$gte": since}},
                    {"timestamp": 1, "focus_score": 1, "_id": 0},
                ).sort("timestamp", 1)
            )
        except Exception as exc:
            logger.warning(
                "focus_trend_query_error",
                extra={"user_id": user_id, "error": str(exc)},
            )
            return None

        if len(docs) < 2:
            return None

        # Convert timestamps to seconds from the first observation
        t0: datetime = docs[0]["timestamp"]
        xs = [(d["timestamp"] - t0).total_seconds() for d in docs]
        ys = [d["focus_score"] for d in docs]

        # OLS slope
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denominator = sum((x - mean_x) ** 2 for x in xs)

        if denominator == 0.0:
            return 0.0

        slope = numerator / denominator
        return round(slope, 6)
