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
- result payload is the JSON-safe CoachAction dump, carrying the strict
  `nudge` CoachOutput (COACH-05) plus a sanitized `coach_error` when the LLM
  output could not be parsed/validated; the Node result consumer correlates
  it back to the AiJob (AI-COM-07)
- `nudge` passes shape + content-policy validation (COACH-06): list-based
  filter with an LLM-guard fallback, one correction retry, then the job FAILS
  terminal with a sanitized reason

COACH-08: LLM timeout/quota failures surface as RetryableError so the shared
retry policy (AI-COM-06) owns recovery; on the final attempt the worker falls
back to the deterministic rule engine (`apply_rules`) and marks the result
`fallbackUsed: true` (mirrors planner_worker PLAN-06). The fallback action
still passes COACH-06 validation before it is returned.

COACH-09: every completed decision (normal or rule-engine fallback) is
persisted to the coach history collection on job completion, idempotently by
`correlationId` — a retried or redelivered job cannot duplicate a history row.
The failure path never reaches persistence, so it leaves no partial records.

COACH-10 moves the Node caller off the legacy HTTP route. COACH-13 will feed
the bounded `signals` window into the coach context (today the live flattened
fields are used).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from messaging.failures import RetryableError, TerminalError, sanitized_error
from messaging.topology import MAX_RETRIES
from pydantic import ValidationError
from utils.logger import get_logger
from workers.base import BaseAIWorker
from workers.schemas import CoachRequest

from agents.coach.decision.output_validator import CoachOutputRejectedError

logger = get_logger(__name__)


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
        self._history_repo = None

    @property
    def orchestrator(self):
        if self._orchestrator is None:
            from services.ai_orchestrator.orchestrator import AIOrchestrator

            self._orchestrator = AIOrchestrator()
        return self._orchestrator

    @property
    def history_repo(self):
        """Lazy CoachHistoryRepository; no-ops when the history store is down."""
        if self._history_repo is None:
            from agents.coach.services.coach_history_repository import (
                CoachHistoryRepository,
            )

            self._history_repo = CoachHistoryRepository()
        return self._history_repo

    async def handle(self, payload: Dict[str, Any], envelope) -> Dict[str, Any]:
        request = self._build_request(payload)
        fallback_used = False
        try:
            action = await asyncio.to_thread(
                self.orchestrator.run_coach,
                user_id=envelope.userId,
                trace_id=envelope.correlationId,
                **request.to_coach_context(),
            )
        except CoachOutputRejectedError as exc:
            # COACH-06: output failed shape/content-policy validation after one
            # correction retry → job FAILED with a sanitized reason.
            raise TerminalError(str(exc)) from exc
        except RetryableError as exc:
            # COACH-08: underneath MAX_RETRIES the shared retry policy owns
            # recovery (NACK + TTL'd delay queues). On the final attempt the
            # job is NOT dead-lettered — it falls back to the deterministic
            # rule engine and still COMPLETES.
            if getattr(self, "current_attempt", 0) < MAX_RETRIES:
                raise
            action = await self._fallback_action(request, envelope.userId, exc)
            fallback_used = True
        # COACH-09 AC#1/#2: persist on job completion, idempotent by
        # correlationId. AC#3: failed jobs never reach this line.
        await self._persist_result(request, envelope, action)
        return self._coach_payload(action, fallback_used=fallback_used)

    # ------------------------------------------------------------ internals

    def _build_request(self, payload: Any) -> CoachRequest:
        if not isinstance(payload, dict):
            raise TerminalError("invalid payload: coach payload must be an object")
        try:
            return CoachRequest.model_validate(payload)
        except ValidationError as exc:
            raise TerminalError(f"invalid coach request: {exc}") from exc

    def _build_coach_input(self, request: CoachRequest):
        """Map a validated CoachRequest onto the CoachInput the rules read."""
        from datetime import datetime, timezone

        from agents.coach.models.schemas import (
            CoachInput,
            FatigueState,
            FocusState,
        )

        current_time = request.current_time or datetime.now(timezone.utc)
        return CoachInput(
            scheduled_tasks=[],
            current_time=current_time,
            focus_state=FocusState(
                state=request.focus_state or "Drifting",
                score=request.focus_score if request.focus_score is not None else 0.5,
            ),
            fatigue_state=FatigueState(
                state=request.fatigue_state or "Moderate",
                score=request.fatigue_score if request.fatigue_score is not None else 0.3,
            ),
            affective_state="engaged",
            ignored_count=request.ignored_count,
            do_not_disturb=request.do_not_disturb,
            is_late=False,
            signals=None,
        )

    async def _fallback_action(self, request, user_id: str, cause: Exception):
        """Final-attempt rule-engine fallback (COACH-08).

        Mirrors planner_worker._fallback_result (PLAN-06): after retries are
        exhausted, produce a deterministic, validated decision instead of
        dead-lettering. The returned CoachAction still passes COACH-06
        validation; `handle` marks the result `fallbackUsed: true` and
        persists it (COACH-09).
        """
        from agents.coach.decision.output_parser import safe_fallback_nudge
        from agents.coach.decision.output_validator import (
            check_coach_output,
            sanitize_nudge,
        )
        from agents.coach.models.schemas import CoachAction
        from agents.coach.rules.rule_engine import apply_rules

        input_data = self._build_coach_input(request)

        action = apply_rules(input_data)
        if action is None:
            # No hard rule matched (the same rules short-circuited inside
            # run_coach, which is why the LLM ran in the first place).
            # Guarantee a decision with the fixed COACH-05 fallback nudge
            # instead of dead-lettering.
            nudge = safe_fallback_nudge()
            action = CoachAction(
                action_type="nudge",
                message=nudge.nudge_text,
                reasoning=(
                    "Coach LLM unavailable after retries; used a safe "
                    "rule-engine fallback nudge."
                ),
                nudge=nudge,
                coach_error=sanitized_error(cause),
            )

        # AC#3 — the fallback decision still passes COACH-05/06 validation.
        if action.nudge is not None:
            action.nudge = sanitize_nudge(action.nudge)
            problems = check_coach_output(action.nudge)
            if problems:
                safe_nudge = safe_fallback_nudge()
                action.nudge = safe_nudge
                action.message = None
                action.coach_error = (
                    "fallback decision rejected by coach validation: "
                    + "; ".join(problems[:3])
                )
        else:
            action.coach_error = action.coach_error or sanitized_error(cause)

        logger.info(
            "coach_rule_fallback_used",
            extra={
                "user_id": user_id,
                "action_type": action.action_type,
                "coach_error": action.coach_error,
            },
        )
        return action

    async def _persist_result(self, request, envelope, action) -> None:
        """COACH-09: persist the completed nudge to coach history.

        Runs off the event loop; the repository degrades to a no-op when the
        history store is unavailable, so persistence can never fail a job that
        already produced a valid decision.
        """
        if action is None or not envelope.userId:
            return
        try:
            input_data = self._build_coach_input(request)
            await asyncio.to_thread(
                self.history_repo.save_action,
                envelope.userId,
                action,
                input_data,
                trace_id=envelope.correlationId,
                correlation_id=envelope.correlationId,
            )
        except Exception as exc:
            logger.warning(
                "coach_persist_result_error",
                extra={
                    "error": sanitized_error(exc),
                    "userId": envelope.userId,
                },
            )

    def _coach_payload(self, action: Any, fallback_used: bool = False) -> Dict[str, Any]:
        from agents.coach.models.schemas import CoachAction

        if isinstance(action, CoachAction):
            result = action.model_dump(mode="json")
        elif isinstance(action, dict):
            try:
                result = CoachAction.model_validate(action).model_dump(mode="json")
            except Exception as exc:
                raise TerminalError(f"coach produced invalid action: {exc}") from exc
        else:
            raise TerminalError("coach produced an invalid action")
        # COACH-08 AC#4: the result carries whether the rule engine produced it.
        result["fallbackUsed"] = bool(fallback_used)
        return result


if __name__ == "__main__":  # pragma: no cover - manual run entrypoint
    logging = __import__("logging")
    logging.basicConfig(level=logging.INFO)
    CoachWorker().run()