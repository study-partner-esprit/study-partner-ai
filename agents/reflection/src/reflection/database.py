"""
Database connection and collections for the Reflection Agent.
"""

import logging
from pymongo import MongoClient, ASCENDING
from pymongo.errors import OperationFailure

from src.config.settings import MONGO_URI, MONGO_DB

logger = logging.getLogger(__name__)

_client: "MongoClient | None" = None
_db = None
_daily_metrics_collection = None
_reflections_collection = None


def get_client() -> MongoClient:
    """Get or create MongoDB client singleton."""
    global _client
    if _client is None:
        from src.config.settings import MONGO_URI
        _client = MongoClient(MONGO_URI)
        logger.info("MongoDB client created")
    return _client


def get_db():
    """Get database instance."""
    global _db
    if _db is None:
        from src.config.settings import MONGO_URI, MONGO_DB
        client = get_client()
        _db = client[MONGO_DB]
    return _db


def get_daily_metrics_collection():
    """Get daily metrics collection with indexes."""
    global _daily_metrics_collection
    if _daily_metrics_collection is None:
        db = get_db()
        _daily_metrics_collection = db["daily_metrics"]
        _create_indexes(_daily_metrics_collection)
    return _daily_metrics_collection


def get_reflections_collection():
    """Get reflections collection."""
    global _reflections_collection
    if _reflections_collection is None:
        db = get_db()
        _reflections_collection = db["reflections"]
    return _reflections_collection


def _create_indexes(collection) -> None:
    """Create necessary indexes for the collection."""
    try:
        collection.create_index(
            [("user_id", 1), ("date", 1)],
            unique=True,
            name="user_date_unique"
        )
        logger.info("Daily metrics indexes created")
    except Exception as e:
        logger.warning(f"Index creation warning: {e}")


def close_connections() -> None:
    """Close database connections."""
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed")