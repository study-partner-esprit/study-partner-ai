"""Signal processing endpoints (focus, fatigue, analyze-frame)."""

import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from deps import (
    _frame_last_request,
    get_focus_detector,
    get_fatigue_detector,
    get_signal_service,
    signals_collection,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class SignalProcessingRequest(BaseModel):
    user_id: str


class CalibrateSignalsRequest(BaseModel):
    user_id: str
    baseline_focus: float = 0.5
    baseline_fatigue: float = 0.2


class FatigueResetRequest(BaseModel):
    user_id: str


@router.get("/api/ai/signals/current/{user_id}")
async def get_current_signals(user_id: str):
    """
    Get the current signal snapshot (focus and fatigue) for a user.

    Args:
        user_id: User identifier

    Returns:
        Latest signal snapshot with focus and fatigue data
    """
    try:
        signal_service = get_signal_service()
        if signal_service is None:
            raise HTTPException(
                status_code=503, detail="Signal processing service is disabled"
            )

        snapshot = signal_service.get_latest_snapshot(user_id)
        if snapshot is None:
            # Generate a new snapshot if none exists
            snapshot = signal_service.get_current_signal_snapshot(user_id)

        return {
            "user_id": user_id,
            "timestamp": snapshot.timestamp,
            "focus": {
                "state": snapshot.focus_state,
                "score": snapshot.focus_score,
                "confidence": snapshot.focus_confidence,
            },
            "fatigue": {
                "state": snapshot.fatigue_state,
                "score": snapshot.fatigue_score,
                "confidence": snapshot.fatigue_confidence,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch signals: {str(e)}"
        )


@router.get("/api/ai/signals/history/{user_id}")
async def get_signal_history(user_id: str, limit: int = 50):
    """Get signal history for a user."""
    try:
        snapshots = get_signal_service().repository.get_signal_history(user_id, limit)
        return {
            "signals": [
                {
                    "timestamp": s.timestamp,
                    "focus": {"state": s.focus_state, "score": s.focus_score},
                    "fatigue": {"state": s.fatigue_state, "score": s.fatigue_score},
                }
                for s in snapshots
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch history: {str(e)}"
        )


@router.post("/api/ai/signals/process")
async def process_signals(request: SignalProcessingRequest):
    """
    Manually trigger signal processing for a user.

    This endpoint would typically be called by a frontend during an active study session.
    """
    try:
        snapshot = get_signal_service().get_current_signal_snapshot(
            user_id=request.user_id,
            video_features=None,  # Frontend should send video data
            video_frame=None,
        )

        return {
            "status": "success",
            "snapshot": {
                "focus": {"state": snapshot.focus_state, "score": snapshot.focus_score},
                "fatigue": {
                    "state": snapshot.fatigue_state,
                    "score": snapshot.fatigue_score,
                },
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Signal processing failed: {str(e)}"
        )


@router.post("/api/ai/signals/analyze-frame")
async def analyze_frame(user_id: str = Form(...), frame: UploadFile = File(...)):
    """
    Analyze a video frame for focus and fatigue detection.

    Args:
        user_id: User ID
        frame: Video frame image file (JPEG/PNG)

    Returns:
        Combined focus and fatigue analysis results
    """
    # --- Per-user timing gate (min 1.5 s between requests) ---
    now = time.time()
    last = _frame_last_request.get(user_id, 0)
    if now - last < 1.5:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit: wait before sending another frame"},
        )
    _frame_last_request[user_id] = now

    try:
        # Read frame data
        frame_data = await frame.read()

        # Run both detectors (per-user fatigue detector for state persistence)
        focus_detector = get_focus_detector()
        fatigue_detector = get_fatigue_detector(user_id)

        focus_result = focus_detector.analyze_frame(frame_data)
        fatigue_result = fatigue_detector.analyze_frame(frame_data)

        # Combine results
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "focus": {
                "score": focus_result.get("focus_score", 0),
                "state": focus_result.get("focus_state", "unknown"),
                "confidence": focus_result.get("confidence", 0),
            },
            "fatigue": {
                "score": fatigue_result.get("fatigue_score", 0),
                "state": fatigue_result.get("fatigue_state", "unknown"),
                "indicators": fatigue_result.get("indicators", {}),
                "confidence": fatigue_result.get("confidence", 0),
            },
        }

        # Save to MongoDB signals collection
        signals_collection.insert_one(analysis)

        return analysis

    except Exception as e:
        logger.warning("frame_analysis_failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Frame analysis failed: {str(e)}")


@router.get("/api/ai/signals/latest/{user_id}")
async def get_latest_signals(user_id: str, limit: int = 10):
    """
    Get the most recent signal analysis results for a user.

    Args:
        user_id: User ID
        limit: Number of results to return

    Returns:
        List of recent signal analyses
    """
    try:
        signals = list(
            signals_collection.find({"user_id": user_id})
            .sort("timestamp", -1)
            .limit(limit)
        )

        # Convert ObjectId to string
        for signal in signals:
            signal["_id"] = str(signal["_id"])

        return {"signals": signals}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch signals: {str(e)}"
        )


@router.post("/api/ai/signals/calibrate")
async def calibrate_signals(request: CalibrateSignalsRequest):
    """
    Reset the EMA state for a user (e.g. at the start of a new study session).
    Optionally seed with known baseline values.
    """
    try:
        from services.signal_processing_service.smoothing import get_ema_state

        ema = get_ema_state()
        ema.reset(request.user_id)
        # Seed with baseline
        ema.update(request.user_id, request.baseline_focus, request.baseline_fatigue)
        return {"status": "ok", "user_id": request.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/ai/signals/fatigue/reset")
async def reset_fatigue(request: FatigueResetRequest):
    """
    Reset the per-user fatigue detector state (blink/yawn counters).
    Call at the start of a new study session.
    """
    try:
        from services.signal_processing_service.fatigue_detector import (
            reset_fatigue_detector,
        )

        reset_fatigue_detector(request.user_id)
        return {"status": "ok", "user_id": request.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
