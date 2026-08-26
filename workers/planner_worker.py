"""PlannerWorker — consumes `study.plan.generate` jobs (F02 / PLAN-01).

Wraps the existing PlannerAgent so plan generation runs inside the RabbitMQ
job bus instead of the synchronous HTTP path:

- payload validated by the strict PlannerRequest schema (PLAN-02); invalid
  input is TERMINAL (retrying cannot fix a malformed request)
- agent.plan() is CPU/RAG-heavy → executed off the event loop via to_thread
- output passes semantic validation before publishing (PLAN-05)
- on the FINAL attempt, LLM outages fall back to SimpleGoalDecomposer and
  still complete with `fallbackUsed: true` (PLAN-06)
- result payload is the JSON-safe PlannerOutput dump; the orchestrator's
  result consumer correlates it back to the AiJob (AI-COM-07)

PLAN-08 moves the Node caller off the legacy HTTP route.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Optional

from messaging.topology import MAX_RETRIES
from messaging.failures import RetryableError, TerminalError
from workers.base import BaseAIWorker
from workers.schemas import PlannerRequest

# Kept for backward-compat with tests; the actual default lives in PlannerRequest
DEFAULT_AVAILABLE_MINUTES = 120


class PlannerWorker(BaseAIWorker):
    job_type = "study.plan.generate"

    def __init__(
        self,
        rabbitmq_url: Optional[str] = None,
        idempotency_store=None,
        prefetch: int = 1,
        agent=None,
    ) -> None:
        super().__init__(
            rabbitmq_url=rabbitmq_url,
            idempotency_store=idempotency_store,
            prefetch=prefetch,
        )
        # PlannerAgent loads embedding models — build lazily on first use so a
        # worker that never handles a job pays no startup cost.
        self._agent = agent

    @property
    def agent(self):
        if self._agent is None:
            from agents.planner.agent import PlannerAgent

            self._agent = PlannerAgent()
        return self._agent

    async def handle(self, payload: Dict[str, Any], envelope) -> Dict[str, Any]:
        from agents.planner.models.task_graph import PlannerInput, TaskGraph, PlannerOutput
        from agents.planner.output_validation import PlanValidationError, validate_plan_output

        request = self._build_request(payload)
        user_id = getattr(envelope, "userId", None) or "anonymous"

        try:
            raw_output = await asyncio.to_thread(
                self.agent.plan, request.to_planner_input(user_id=user_id)
            )
        except RetryableError as exc:
            if getattr(self, "current_attempt", 0) >= MAX_RETRIES:
                return await self._fallback_result(request, exc)
            raise

        try:
            output = (
                raw_output
                if isinstance(raw_output, PlannerOutput)
                else PlannerOutput.model_validate(raw_output)
            )
        except Exception as exc:
            raise TerminalError(f"planner produced invalid output shape: {exc}") from exc

        problems = validate_plan_output(output, available_minutes=request.available_minutes)
        if problems:
            raise TerminalError(f"plan rejected by validation: {'; '.join(problems[:5])}")

        result = output.model_dump(mode="json")
        result["fallbackUsed"] = False
        return result

    # ------------------------------------------------------------ internals

    async def _fallback_result(self, request, cause: Exception) -> Dict[str, Any]:
        """Final-attempt deterministic decomposition (PLAN-06)."""
        from agents.planner.models.task_graph import PlannerOutput, TaskGraph
        from agents.planner.output_validation import validate_plan_output

        loop = asyncio.get_running_loop()
        tasks = await loop.run_in_executor(
            None,
            lambda: self.agent.simple_decomposer.decompose(
                request.goal, request.concepts or [], request.available_minutes
            ),
        )
        graph = TaskGraph(goal=request.goal, tasks=tasks)
        for i, task in enumerate(tasks):  # decomposer may leave ids unset
            if not task.id:
                task.id = f"task-{uuid.uuid4()}"
        output = PlannerOutput(
            task_graph=graph,
            warning=f"LLM planner unavailable ({cause}); used simplified decomposition.",
        )
        problems = validate_plan_output(output, available_minutes=request.available_minutes)
        if problems:
            raise TerminalError(
                f"fallback plan rejected by validation: {'; '.join(problems[:5])}"
            ) from cause
        result = output.model_dump(mode="json")
        result["fallbackUsed"] = True
        return result

    def _build_request(self, payload: Any) -> PlannerRequest:
        from pydantic import ValidationError

        if not isinstance(payload, dict):
            raise TerminalError("plan payload must be an object")
        body = dict(payload)
        body.setdefault("goal", "")
        # Accept legacy field spellings from the pre-bus API.
        if "available_minutes" not in body and "availableTimeMinutes" in body:
            body["available_minutes"] = body["availableTimeMinutes"]
        if "course_id" not in body and "courseId" in body:
            body["course_id"] = body["courseId"]
        if not body.get("goal"):
            # Course-derived plans may omit goal only if course docs provided;
            # strict schema requires goal, so synthesize from course id.
            if body.get("course_id"):
                body["goal"] = f"Complete course {body['course_id']}"
            else:
                raise TerminalError("plan requires 'goal'")
        try:
            return PlannerRequest(**body)
        except ValidationError as exc:
            raise TerminalError(f"invalid planner request: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover - manual run entrypoint
    logging = __import__("logging")
    logging.basicConfig(level=logging.INFO)
    PlannerWorker().run()
