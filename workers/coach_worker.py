"""CoachWorker — consumes `study.coach.nudge` jobs (F03 / COACH-01).

Wraps the existing AIOrchestrator coach pipeline so coaching decisions run
inside the RabbitMQ job bus instead of the synchronous HTTP path:

- envelope validated by BaseAIWorker (AI-COM-02); identity always taken from
  `envelope.userId`, never from the payload body
- the payload is validated by the bounded CoachRequest schema (COACH-02):
  session context (session_id, recent signals, recent chat messages) is size
  capped and limits-bound before anything reaches the LLM; malformed input is
  TERMINAL (retrying cannot fix a bad request)
- AIOrchestrator.run_coach is LLM/DB/ML-heavy → executed off the event loop
  via to_thread
- result payload is the JSON-safe CoachAction dump; the Node result consumer
  correlates it back to the AiJob (AI-COM-07)

The strict CoachOutput schema lands with COACH-05. COACH-10 moves the Node
caller off the legacy HTTP route. COACH-13 will feed the bounded `signals`
window into the coach context (today the live flattened fields are used).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from messaging.failures import TerminalError
from pydantic import ValidationError
from workers.base import BaseAIWorker
from workers.schemas import CoachRequest


class CoachWorker(BaseAIWorker):
    job_type = "study.coach.nudge"

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
        # AIOrchestrator loads ML adapters + Mongo clients — build lazily on
        # first use so a worker that never handles a job pays no startup cost.
        self._orchestrator = orchestrator

    @property
    def orchestrator(self):
        if self._orchestrator is None:
            from services.ai_orchestrator.orchestrator import AIOrchestrator

            self._orchestrator = AIOrchestrator()
        return self._orchestrator

    async def handle(self, payload: Dict[str, Any], envelope) -> Dict[str, Any]:
        request = self._build_request(payload)
        action = await asyncio.to_thread(
            self.orchestrator.run_coach,
            user_id=envelope.userId,
            trace_id=envelope.correlationId,
            **request.to_coach_context(),
        )
        return self._coach_payload(action)

    # ------------------------------------------------------------ internals

    def _build_request(self, payload: Any) -> CoachRequest:
        if not isinstance(payload, dict):
            raise TerminalError("invalid payload: coach payload must be an object")
        try:
            return CoachRequest.model_validate(payload)
        except ValidationError as exc:
            raise TerminalError(f"invalid coach request: {exc}") from exc

    def _coach_payload(self, action: Any) -> Dict[str, Any]:
        from agents.coach.models.schemas import CoachAction

        if isinstance(action, CoachAction):
            return action.model_dump(mode="json")
        if isinstance(action, dict):
            try:
                return CoachAction.model_validate(action).model_dump(mode="json")
            except Exception as exc:
                raise TerminalError(f"coach produced invalid action: {exc}") from exc
        raise TerminalError("coach produced an invalid action")


if __name__ == "__main__":  # pragma: no cover - manual run entrypoint
    logging = __import__("logging")
    logging.basicConfig(level=logging.INFO)
    CoachWorker().run()