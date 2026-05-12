"""
MongoDB client for connecting to the study_partner database.
"""

import logging
from typing import Optional

try:
    from pymongo import MongoClient
    from pymongo.database import Database
except ImportError:
    raise ImportError("pymongo required. Install with: pip install pymongo")

from src.config.settings import MONGO_URI, DATABASE_NAME

logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB client for the evaluator project."""

    def __init__(self, uri: str = MONGO_URI, database_name: str = DATABASE_NAME):
        self.uri = uri
        self.database_name = database_name
        self._client: Optional[MongoClient] = None
        self._database: Optional[Database] = None

    def connect(self) -> Database:
        """Connect to MongoDB and return the database."""
        if self._database is not None:
            return self._database

        try:
            self._client = MongoClient(self.uri)
            self._database = self._client[self.database_name]
            logger.info(f"Connected to MongoDB database: {self.database_name}")
            return self._database
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def close(self):
        """Close the MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("MongoDB connection closed")

    @property
    def db(self) -> Database:
        """Get the database instance. Connects if not already connected."""
        if self._database is None:
            return self.connect()
        return self._database