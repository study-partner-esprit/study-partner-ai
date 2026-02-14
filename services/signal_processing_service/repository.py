"""Repository for persisting and retrieving signal snapshots from MongoDB.

This module handles all database operations for user signals.
"""

from pymongo import MongoClient, DESCENDING
from typing import Optional
from datetime import datetime
import os

from services.signal_processing_service.signal_snapshot import SignalSnapshot


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
            {"user_id": user_id},
            sort=[("_id", DESCENDING)]
        )
        
        if document is None:
            return None
        
        # Remove MongoDB's _id field before parsing
        document.pop("_id", None)
        
        return SignalSnapshot(**document)
    
    def get_signal_history(
        self, 
        user_id: str, 
        limit: int = 10
    ) -> list[SignalSnapshot]:
        """
        Retrieve recent signal history for a user.
        
        Args:
            user_id: The user's unique identifier
            limit: Maximum number of snapshots to return
            
        Returns:
            List of SignalSnapshot objects, ordered by timestamp (newest first)
        """
        documents = self.collection.find(
            {"user_id": user_id},
            sort=[("timestamp", DESCENDING)],
            limit=limit
        )
        
        snapshots = []
        for doc in documents:
            doc.pop("_id", None)
            snapshots.append(SignalSnapshot(**doc))
        
        return snapshots
