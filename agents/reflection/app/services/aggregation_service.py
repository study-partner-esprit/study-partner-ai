from app.database import daily_metrics_collection
from pymongo.errors import PyMongoError
from datetime import date, timedelta


def get_weekly_summary(user_id: str, reference_date: date = None) -> dict:
    """
    Calcule le résumé de la semaine contenant reference_date.
    Par défaut : la semaine en cours.
    """
    if reference_date is None:
        reference_date = date.today()

    # Calcule le lundi et dimanche de la semaine
    start_of_week = reference_date - timedelta(days=reference_date.weekday())
    end_of_week   = start_of_week + timedelta(days=6)

    pipeline = [
        # Étape 1 — Filtrer par user et par plage de dates
        {
            "$match": {
                "user_id": user_id,
                "date": {
                    "$gte": str(start_of_week),
                    "$lte": str(end_of_week)
                }
            }
        },
        # Étape 2 — Grouper et calculer les agrégats
        {
            "$group": {
                "_id": "$user_id",
                "total_study_minutes": {"$sum": "$total_study_minutes"},
                "total_xp_earned":     {"$sum": "$xp_earned"},
                "total_sessions":      {"$sum": "$sessions_count"},
                "avg_focus_score":     {"$avg": "$avg_focus_score"},
                "avg_fatigue_score":   {"$avg": "$avg_fatigue_score"},
                "days_studied":        {"$sum": 1},
                "best_focus_day":      {"$max": "$avg_focus_score"},
                "worst_fatigue_day":   {"$max": "$avg_fatigue_score"},
            }
        },
        # Étape 3 — Reformater la réponse proprement
        {
            "$project": {
                "_id": 0,
                "user_id":              "$_id",
                "week_start":           {"$literal": str(start_of_week)},
                "week_end":             {"$literal": str(end_of_week)},
                "total_study_minutes":  1,
                "total_xp_earned":      1,
                "total_sessions":       1,
                "days_studied":         1,
                "avg_focus_score":      {"$round": ["$avg_focus_score", 2]},
                "avg_fatigue_score":    {"$round": ["$avg_fatigue_score", 2]},
                "best_focus_day":       {"$round": ["$best_focus_day", 2]},
                "worst_fatigue_day":    {"$round": ["$worst_fatigue_day", 2]},
            }
        }
    ]

    try:
        results = list(daily_metrics_collection.aggregate(pipeline))
        if not results:
            return {
                "user_id":    user_id,
                "week_start": str(start_of_week),
                "week_end":   str(end_of_week),
                "message":    "No data found for this week"
            }
        return results[0]

    except PyMongoError as e:
        return {"status": "error", "detail": str(e)}


def get_all_weeks_summary(user_id: str) -> list:
    """
    Calcule un résumé agrégé par semaine pour tout l'historique d'un user.
    """
    pipeline = [
        {"$match": {"user_id": user_id}},

        # Convertir la date string en vrai objet date pour extraire la semaine
        {
            "$addFields": {
                "date_obj": {"$dateFromString": {"dateString": "$date"}}
            }
        },

        # Grouper par semaine ISO (année + numéro de semaine)
        {
            "$group": {
                "_id": {
                    "year": {"$isoWeekYear": "$date_obj"},
                    "week": {"$isoWeek":     "$date_obj"}
                },
                "total_study_minutes": {"$sum": "$total_study_minutes"},
                "total_xp_earned":     {"$sum": "$xp_earned"},
                "total_sessions":      {"$sum": "$sessions_count"},
                "avg_focus_score":     {"$avg": "$avg_focus_score"},
                "avg_fatigue_score":   {"$avg": "$avg_fatigue_score"},
                "days_studied":        {"$sum": 1},
            }
        },

        # Trier chronologiquement
        {"$sort": {"_id.year": 1, "_id.week": 1}},

        # Reformater
        {
            "$project": {
                "_id": 0,
                "year":                "$_id.year",
                "week":                "$_id.week",
                "total_study_minutes": 1,
                "total_xp_earned":     1,
                "total_sessions":      1,
                "days_studied":        1,
                "avg_focus_score":     {"$round": ["$avg_focus_score", 2]},
                "avg_fatigue_score":   {"$round": ["$avg_fatigue_score", 2]},
            }
        }
    ]

    try:
        return list(daily_metrics_collection.aggregate(pipeline))
    except PyMongoError as e:
        return [{"status": "error", "detail": str(e)}]