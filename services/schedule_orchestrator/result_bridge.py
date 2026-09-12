"""Correlation bridge for chained AI jobs (F03 / COACH-16).

The CoachWorker publishes a `study.schedule.apply` downstream job and must
report whether the schedule was actually applied (AC#4: never silent). We
resolve that with a module-level in-process `asyncio.Future` keyed by
correlationId:

- `await_schedule_cast(change, user_id, trace_id)` registers a Future and
  returns `(status, detail)` for the await call.
- `register_pending(correlation_id, reason)` records that a schedule apply
  started under a correlationId.
- `resolve(correlation_id, status, detail)` is called by `BaseAIWorker` when
  the downstream worker's result event is published: the Future is set and
  the registered entry records the outcome.
- `snapshot(correlation_id)` returns the recorded outcome (used by the coach
  to surface `schedule_update.status`).

Both workers share one process in this deploy, so the bridge works end-to-end.
If the downstream worker cannot be reached (no registered producer, timeout,
or publish failure), the coach reports `status="error"` — never silent. The
bridged failure degrades gracefully and is best-effort: it can never raise
from a result publication.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_locked: threading.Lock = threading.Lock()
_pending: Dict[str, str] = {}        # correlationId → reason
_outcomes: Dict[str, Dict[str, Any]] = {}  # correlationId → {status, detail}
_waiters: Dict[str, asyncio.Future] = {}

DEFAULT_AWAIT_TIMEOUT_S = float(__import__("os").getenv(
    "COACH_SCHEDULE_AWAIT_TIMEOUT_S", "5"
))


def register_pending(correlation_id: str, reason: str) -> None:
    with _locked:
        _pending[correlation_id] = reason


def snapshot(correlation_id: str) -> Optional[Dict[str, Any]]:
    with _locked:
        return _outcomes.get(correlation_id)


def resolve(correlation_id: str, status: str, detail: Optional[str] = None) -> None:
    """Record a settled outcome for a correlationId; resolve any waiter."""
    with _locked:
        _pending.pop(correlation_id, None)
        _outcomes[correlation_id] = {"status": status, "detail": detail}
    # Settle an awaiting Future if one is registered (best-effort).
    fut = _waiters.get(correlation_id)
    if fut is not None and not fut.done():
        fut.set_result({"status": status, "detail": detail})


async def await_schedule_cast(
    correlation_id: str,
    reason: str,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Wait for the downstream apply result under `correlation_id`.

    Returns `{status, detail}`. On timeout returns `{status: "error"}` —
    callers must never let an unresolved cast fail the coach job (AC#4: the
    apply error is surfaced as `schedule_update.status`, not thrown).
    """
    timeout = timeout_s if timeout_s is not None else DEFAULT_AWAIT_TIMEOUT_S
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    with _locked:
        _waiters[correlation_id] = fut
    register_pending(correlation_id, reason)
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "schedule_cast_timeout",
            extra={"correlationId": correlation_id, "reason": reason},
        )
        return {"status": "error", "detail": "apply timed out"}
    finally:
        with _locked:
            _waiters.pop(correlation_id, None)
