"""
Reflection Agent FastAPI application.
"""

import logging
from datetime import date
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from src.config.settings import API_HOST, API_PORT, LOG_LEVEL
from src.reflection.database import close_connections
from src.reflection.schemas import (
    DailyMetrics,
    WeeklySummary,
    TrendResponse,
    ReflectionResponse,
    ErrorResponse,
)
from src.reflection.services import (
    upsert_daily_metrics,
    get_weekly_summary,
    get_all_weeks_summary,
    compute_trends,
    generate_reflection,
    get_user_reflections,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Reflection Agent starting up...")
    yield
    logger.info("Reflection Agent shutting down...")
    close_connections()


app = FastAPI(
    title="Reflection Agent",
    version="2.0.0",
    description="Weekly reflection generation and study analytics",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Reflection Agent running — v2.0", "status": "healthy"}


@app.post("/simulate-day", response_model=dict, responses={500: {"model": ErrorResponse}})
async def simulate_day(data: DailyMetrics):
    """
    Record or update daily study metrics for a user.
    """
    result = upsert_daily_metrics(data.model_dump())
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["detail"])
    return result


@app.get("/analytics/{user_id}/weekly", response_model=dict, responses={500: {"model": ErrorResponse}})
async def weekly_summary(user_id: str, reference_date: date = None):
    """
    Get weekly summary for the week containing reference_date.
    Defaults to current week.
    """
    result = get_weekly_summary(user_id, reference_date)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result["detail"])
    return result


@app.get("/analytics/{user_id}/history", response_model=list)
async def full_history(user_id: str):
    """
    Get all aggregated weekly summaries for a user.
    """
    result = get_all_weeks_summary(user_id)
    if result and isinstance(result[0], dict) and result[0].get("status") == "error":
        raise HTTPException(status_code=500, detail=result[0]["detail"])
    return result


@app.get("/analytics/{user_id}/trends", response_model=dict, responses={500: {"model": ErrorResponse}})
async def user_trends(user_id: str):
    """
    Analyze trends week over week.
    Returns progression score and detailed trends.
    """
    result = compute_trends(user_id)
    if result.get("status") in ("error", "insufficient_data"):
        raise HTTPException(status_code=400 if result.get("status") == "insufficient_data" else 500, detail=result.get("detail", "Unknown error"))
    return result


@app.get("/analytics/{user_id}/reflection", response_model=dict, responses={500: {"model": ErrorResponse}})
async def user_reflection(user_id: str):
    """
    Generate and store an intelligent weekly reflection based on trends.
    """
    result = generate_reflection(user_id)
    if result.get("status") in ("error", "insufficient_data"):
        raise HTTPException(status_code=400 if result.get("status") == "insufficient_data" else 500, detail=result.get("detail", "Unknown error"))
    return result


@app.get("/analytics/{user_id}/reflections", response_model=list)
async def user_reflections(user_id: str, limit: int = 10):
    """
    Retrieve stored reflections for a user.
    """
    from src.reflection.services.reflection_service import get_user_reflections
    return get_user_reflections(user_id, limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.reflection.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )