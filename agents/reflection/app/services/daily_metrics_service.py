from app.database import daily_metrics_collection
from pymongo.errors import PyMongoError

def upsert_daily_metrics(data: dict) -> dict:
    filter_key = {
        "user_id": data["user_id"],
        "date": str(data["date"])
    }

    update_payload = {
        "$set": {
            "total_study_minutes": data["total_study_minutes"],
            "avg_focus_score":     data["avg_focus_score"],
            "avg_fatigue_score":   data["avg_fatigue_score"],
            "xp_earned":           data["xp_earned"],
            "sessions_count":      data["sessions_count"]
        }
    }

    try:
        result = daily_metrics_collection.update_one(
            filter_key,
            update_payload,
            upsert=True
        )
        if result.upserted_id:
            return {"status": "created", "id": str(result.upserted_id)}
        else:
            return {"status": "updated", "matched": result.matched_count}

    except PyMongoError as e:
        return {"status": "error", "detail": str(e)}