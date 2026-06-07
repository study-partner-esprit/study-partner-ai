from pymongo import MongoClient, ASCENDING
from pymongo.errors import OperationFailure
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["reflection_db"]

daily_metrics_collection = db["daily_metrics"]
reflections_collection = db["reflections"]

def create_indexes():
    try:
        daily_metrics_collection.create_index(
            [("user_id", ASCENDING), ("date", ASCENDING)],
            unique=True,
            name="user_date_unique"
        )
        print("Indexes created successfully")
    except OperationFailure as e:
        print(f"Index issue: {e}")

create_indexes()