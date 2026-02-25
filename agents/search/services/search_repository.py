"""SearchRepository — persist and retrieve search exchanges from MongoDB.

Collection: ``search_history``

Document schema::

    {
        user_id:    str,
        session_id: str,
        trace_id:   str,
        ts:         datetime (UTC),
        question:   str,
        answer:     str,
        sources:    [str],   # URLs used to produce the answer
    }
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("DB_NAME", "study_partner")
COLLECTION = "search_history"


class SearchRepository:
    """Thin wrapper around the ``search_history`` MongoDB collection."""

    def __init__(self) -> None:
        self._col = None
        try:
            from pymongo import MongoClient
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            self._col = client[DB_NAME][COLLECTION]
        except Exception as exc:
            logger.warning("search_repo_no_db", extra={"error": str(exc)})

    # ------------------------------------------------------------------ #

    def save_exchange(
        self,
        user_id: str,
        question: str,
        answer: str,
        sources: Optional[List[str]] = None,
        session_id: str = "",
        trace_id: str = "",
    ) -> None:
        if self._col is None or not user_id:
            return
        try:
            self._col.insert_one({
                "user_id":    user_id,
                "session_id": session_id,
                "trace_id":   trace_id,
                "ts":         datetime.now(tz=timezone.utc),
                "question":   question,
                "answer":     answer,
                "sources":    sources or [],
            })
        except Exception as exc:
            logger.warning("search_repo_save_error", extra={"error": str(exc)})

    def get_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        if self._col is None or not user_id:
            return []
        try:
            cursor = (
                self._col
                .find({"user_id": user_id}, {"_id": 0})
                .sort("ts", -1)
                .limit(limit)
            )
            return list(cursor)
        except Exception as exc:
            logger.warning("search_repo_get_error", extra={"error": str(exc)})
            return []

    def clear_history(self, user_id: str) -> None:
        if self._col is None or not user_id:
            return
        try:
            self._col.delete_many({"user_id": user_id})
        except Exception as exc:
            logger.warning("search_repo_clear_error", extra={"error": str(exc)})
