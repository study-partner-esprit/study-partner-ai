"""EvaluatorWorker — consumes `study.eval.step` jobs (F04 / EVAL-01).

Wraps the multi-turn EvaluatorAgent so a single evaluation step runs inside the
RabbitMQ job bus instead of the synchronous HTTP path:

- envelope validated by BaseAIWorker (AI-COM-02); identity from `envelope.userId`
- the agent's multi-turn Socratic state machine is retained end-to-end
  (WHAT → WHY → HOW depth escalation, deterministic mastery scoring, guessing
  detection, max-attempts termination) — this worker only drives one step per
  job and returns the agent's JSON-safe result dict
- the agent is LLM-heavy → executed off the event loop via to_thread
- malformed payloads are TERMINAL (retrying cannot fix a bad request)
- when a `session_id` is supplied the job resumes that session's step;
  otherwise a fresh session is started for the task details

The strict request contract (``EvaluationRequest``: sessionId, step,
student_answer, context_id), session rehydration from a persisted state store,
and the output schema are introduced by EVAL-02 / EVAL-06 respectively.

EVAL-01 removes the direct HTTP exposure for evaluation (the legacy
`/api/ai/evaluator/*` routes are gone).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from messaging.failures import TerminalError
from workers.base import BaseAIWorker


class EvaluatorWorker(BaseAIWorker):
    job_type = "study.eval.step"

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
        # EvaluatorAgent builds a LiteLLM shim lazily — construct on first use
        # so a worker that never handles a job pays no startup cost.
        self._agent = agent

    @property
    def agent(self):
        if self._agent is None:
            from agents.evaluator.agent import EvaluatorAgent

            self._agent = EvaluatorAgent()
        return self._agent

    async def handle(self, payload: Dict[str, Any], envelope) -> Dict[str, Any]:
        request = self._build_request(payload)
        # Evaluation is LLM + scoring heavy → keep it off the event loop.
        return await asyncio.to_thread(self._run_step, request)

    # ------------------------------------------------------------ internals

    def _build_request(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise TerminalError("invalid payload: eval payload must be an object")
        return payload

    def _run_step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = payload.get("session_id")
        if session_id:
            student_answer = payload.get("student_answer")
            if not isinstance(student_answer, str) or not student_answer:
                raise TerminalError("invalid eval step: student_answer required to resume a session")
            # Agent/LLM failures propagate → classified retryable by defaults.
            return self.agent.handle_user_answer(session_id, student_answer)

        task_title = payload.get("task_title", "")
        task_description = payload.get("task_description", "")
        task_details = payload.get("task_details", "")
        if not task_details:
            raise TerminalError("invalid eval step: task_details required to start a session")
        max_attempts = payload.get("max_attempts", 5)
        return self.agent.start_session(
            task_title=task_title,
            task_description=task_description,
            task_details=task_details,
            max_attempts=int(max_attempts),
        )


if __name__ == "__main__":  # pragma: no cover - manual run entrypoint
    import logging

    logging.basicConfig(level=logging.INFO)
    EvaluatorWorker().run()
