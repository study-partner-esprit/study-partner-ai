"""ScheduleWorker — consumes `study.schedule.apply` jobs (F03 / COACH-16).

The CoachWorker, when a coach decision carries `schedule_changes`, publishes a
`study.schedule.apply` job into the bus instead of calling HTTP or mutating
the schedule synchronously. This worker:

- validates the bounded `ScheduleApplyRequest` (coach payload → action,
  duration, new_start_time, affected_task_ids, reasoning)
- applies the change transactionally through `ScheduleOrchestrator`
  (Mongo sessions/transactions across task_scheduling + schedule_history +
  schedule_snapshots) — AC#2: the worker path, never direct HTTP
- is idempotent by correlationId at the bus level (AI-COM-08 messageId) and
  records every change in `schedule_history` inside the same transaction
- returns a result payload whose `schedule_update.status` is `success` on
  completion, `no_changes` when nothing to do, or `error` when the apply
  failed — AC#4: a reschedule failure surfaces as `error`, never silent

The coach worker awaits this result via the in-process correlation bridge,
so its own result reflects the true apply outcome.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import ValidationError

from agents.coach.models.schemas import ScheduleChange
from messaging.failures import TerminalError
from utils.logger import get_logger
from workers.base import BaseAIWorker
from workers.schemas import ScheduleApplyRequest

logger = get_logger(__name__)


class ScheduleWorker(BaseAIWorker):
    job_type = "study.schedule.apply"

    def __init__(
        self,
        rabbitmq_url: Optional[str] = None,
        idempotency_store=None,
        prefetch: int = 1,
        orchestrator=None,
    ) -> None:
        super().__init__(
            rabbitmq_url=rabbitmq_url,
            idempotency_store=idempotency_store,
            prefetch=prefetch,
        )
        self._orchestrator = orchestrator

    @property
    def schedule_orchestrator(self):
        if self._orchestrator is None:
            from services.schedule_orchestrator.orchestrator import ScheduleOrchestrator

            self._orchestrator = ScheduleOrchestrator()
        return self._orchestrator

    async def handle(
        self, payload: Dict[str, Any], envelope
    ) -> Dict[str, Any]:
        request = self._build_request(payload)
        if request.new_start_time is not None:
            current_time = request.new_start_time
        else:
            current_time = datetime.now(timezone.utc)

        change = ScheduleChange(
            action=request.action,
            duration_minutes=request.duration_minutes,
            new_start_time=request.new_start_time,
            affected_task_ids=list(request.affected_task_ids),
            reasoning=request.reasoning,
        )

        result = await asyncio.to_thread(
            self.schedule_orchestrator.process_schedule_change,
            change,
            envelope.userId,
            current_time,
        )

        status = result.get("status", "error")
        if status not in {"success", "no_changes", "error"}:
            status = "error"

        logger.info(
            "schedule_apply_done",
            extra={
                "userId": envelope.userId,
                "correlationId": envelope.correlationId,
                "action": request.action,
                "status": status,
            },
        )
        return {
            "schedule_update": {
                "status": status,
                "message": result.get("message"),
                "action": request.action,
            }
        }

    # ------------------------------------------------------------ internals

    def _build_request(self, payload: Any) -> ScheduleApplyRequest:
        if not isinstance(payload, dict):
            raise TerminalError("invalid payload: schedule apply payload must be an object")
        try:
            return ScheduleApplyRequest.model_validate(payload)
        except ValidationError as exc:
            raise TerminalError(f"invalid schedule apply request: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover - manual run entrypoint
    logging = __import__("logging")
    logging.basicConfig(level=logging.INFO)
    ScheduleWorker().run()
