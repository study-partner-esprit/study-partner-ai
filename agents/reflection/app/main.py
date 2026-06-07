from fastapi import FastAPI, HTTPException
from datetime import date
from app.schemas import DailyMetrics
from app.services.daily_metrics_service import upsert_daily_metrics
from app.services.aggregation_service import get_weekly_summary, get_all_weeks_summary
from app.services.trend_service import compute_trends
from app.services.reflection_service import generate_reflection
app = FastAPI(title="Reflection Service", version="2.0.0")


@app.get("/")
def root():
    return {"message": "Reflection Service running — v2.0"}


@app.post("/simulate-day")
def simulate_day(data: DailyMetrics):
    result = upsert_daily_metrics(data.model_dump())
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["detail"])
    return result


@app.get("/analytics/{user_id}/weekly")
def weekly_summary(user_id: str, reference_date: date = None):
    """
    Résumé de la semaine contenant reference_date.
    Exemple : /analytics/user-001/weekly?reference_date=2026-04-10
    """
    result = get_weekly_summary(user_id, reference_date)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["detail"])
    return result


@app.get("/analytics/{user_id}/history")
def full_history(user_id: str):
    """
    Toutes les semaines agrégées pour un utilisateur.
    Exemple : /analytics/user-001/history
    """
    result = get_all_weeks_summary(user_id)
    return result
# Ajoute ces deux endpoints à la fin du fichier
@app.get("/analytics/{user_id}/trends")
def user_trends(user_id: str):
    """
    Analyse les tendances semaine sur semaine.
    Exemple : /analytics/user-001/trends
    """
    result = compute_trends(user_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["detail"])
    return result


@app.get("/analytics/{user_id}/reflection")
def user_reflection(user_id: str):
    """
    Génère et stocke une réflexion intelligente basée sur les tendances.
    Exemple : /analytics/user-001/reflection
    """
    result = generate_reflection(user_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["detail"])
    return result