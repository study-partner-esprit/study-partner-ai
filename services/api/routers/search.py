"""AI search endpoints."""

import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class SearchAskRequest(BaseModel):
    question: str
    user_id: Optional[str] = ""
    session_id: Optional[str] = ""


@router.post("/api/ai/search/ask")
async def search_ask(
    req: SearchAskRequest,
    x_trace_id: Optional[str] = Header(None, alias="x-trace-id"),
):
    """Web-search a question using Apify + extract text + answer via LM Studio (Qwen)."""
    trace_id = x_trace_id or str(uuid.uuid4())
    try:
        from agents.search.agent import process_question

        result = process_question(
            req.question,
            user_id=req.user_id,
            session_id=req.session_id,
            trace_id=trace_id,
        )
        if not result.get("success") and result.get("error") == "No question provided":
            raise HTTPException(status_code=400, detail="No question provided")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            "search_ask_error", extra={"error": str(exc), "trace_id": trace_id}
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/ai/search/history/{user_id}")
async def search_history(user_id: str, limit: int = 20):
    """Return the last N search exchanges for a user."""
    try:
        from agents.search.services.search_repository import SearchRepository

        repo = SearchRepository()
        items = repo.get_history(user_id, limit=limit)
        return {"user_id": user_id, "history": items}
    except Exception as exc:
        logger.warning("search_history_error", extra={"error": str(exc)})
        return {"user_id": user_id, "history": []}


@router.delete("/api/ai/search/history/{user_id}")
async def search_history_clear(user_id: str):
    """Clear all search history for a user."""
    try:
        from agents.search.services.search_repository import SearchRepository

        repo = SearchRepository()
        repo.clear_history(user_id)
        return {"success": True}
    except Exception as exc:
        logger.warning("search_history_clear_error", extra={"error": str(exc)})
        return {"success": False, "error": str(exc)}
