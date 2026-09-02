"""EvaluatorWorker — consumes `study.eval.step` jobs (F04 / EVAL-01, EVAL-02).

Wraps the multi-turn EvaluatorAgent so a single evaluation step runs inside the
RabbitMQ job bus instead of the synchronous HTTP path:

- envelope validated by BaseAIWorker (AI-COM-02); identity from `envelope.userId`
- the agent's multi-turn Socratic state machine is retained end-to-end
- the agent is LLM-heavy → executed off the event loop via to_thread
- malformed payloads are TERMINAL (retrying cannot fix a bad request)

EVAL-02 contract (wire is camelCase to match the Node edge validator in
`payloadSchemas.js`): every job carries ``sessionId``, ``step`` (int),
``contextId`` and ``studentAnswer``. ``step == 1`` lazily creates the session
from ``contextId`` (via the context resolver) and processes the first answer;
``step > 1`` resumes that session. After each step the session is serialized to
the session store so multi-turn state survives a worker restart
(rehydration). ``userId`` is only ever taken from the authenticated envelope.

`objectiveId` / Bloom learning-objective targeting (F14) is deferred to a
separate story; EVAL-08 only accepts an OPTIONAL `objectiveId` and carries it
through (with the rest of the per-step record) so the Node backend can persist
it when present.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from pydantic import ValidationError

from agents.evaluator.schemas import EvaluationSession
from messaging.failures import TerminalError
from workers.base import BaseAIWorker
from workers.schemas import EvaluationRequest
from workers.session_store import build_default_session_store


class EvaluatorWorker(BaseAIWorker):
    job_type = "study.eval.step"

    def __init__(
        self,
        rabbitmq_url: Optional[str] = None,
        idempotency_store=None,
        prefetch: int = 1,
        agent=None,
        session_store=None,
        context_resolver=None,
    ) -> None:
        super().__init__(
            rabbitmq_url=rabbitmq_url,
            idempotency_store=idempotency_store,
            prefetch=prefetch,
        )
        # EvaluatorAgent builds a LiteLLM shim lazily — construct on first use
        # so a worker that never handles a job pays no startup cost.
        self._agent = agent
        self.session_store = session_store or build_default_session_store()
        # context_id -> {task_title, task_description, task_details}; supplied
        # by the backend once it resolves context; None uses a minimal default.
        self.context_resolver = context_resolver

    @property
    def agent(self):
        if self._agent is None:
            from agents.evaluator.agent import EvaluatorAgent

            self._agent = EvaluatorAgent()
        return self._agent

    async def handle(self, payload: Dict[str, Any], envelope) -> Dict[str, Any]:
        request = self._build_request(payload)
        # Hydrate a resumed session before the background step so a restart
        # (agent no longer holding the session) can continue its multi-turn run.
        if request.step > 1:
            await self._hydrate(request.session_id)
        # Evaluation is LLM + scoring heavy → keep it off the event loop.
        result = await asyncio.to_thread(self._run_step, request)
        # EVAL-08: enrich the result into a self-contained per-step record (the
        # ai.results payload the Node backend persists to Mongo). Additive only.
        result = self._with_step_record(result, request)
        # Persist the (possibly evolved) session so it survives a restart.
        await self._persist(request.session_id)
        return result

    # ------------------------------------------------------------ internals

    def _build_request(self, payload: Any) -> EvaluationRequest:
        if not isinstance(payload, dict):
            raise TerminalError("invalid eval payload: payload must be an object")
        try:
            return EvaluationRequest.model_validate(payload)
        except ValidationError as e:
            raise TerminalError(f"invalid eval payload: {e}") from e

    async def _hydrate(self, session_id: str) -> None:
        if self.agent.get_session(session_id) is not None:
            return
        serialized = await self.session_store.get(session_id)
        if serialized is None:
            return
        self._restore(session_id, serialized)

    def _restore(self, session_id: str, serialized: str) -> None:
        session = EvaluationSession.model_validate_json(serialized)
        self.agent.restore_session(session)

    async def _persist(self, session_id: str) -> None:
        session = self.agent.get_session(session_id)
        if session is None:
            return
        await self.session_store.put(session_id, session.model_dump_json())

    def _with_step_record(
        self, result: Dict[str, Any], request: EvaluationRequest
    ) -> Dict[str, Any]:
        """EVAL-08: lift a self-contained per-step record onto the result.

        The ai.results payload is the ONLY path to MongoDB — the Python backend
        never writes to Mongo; the Node backend persists whatever arrives here.
        This helper makes that payload a complete, queryable per-step record:
        ``sessionId`` / ``step`` at the top level, plus ``demonstratedBloomLevel``
        (raw feed for BLOOM-08) and ``objectiveId`` when present. Additive only,
        so existing consumers (AiJob.completeByCorrelation) are unaffected.
        """
        if not isinstance(result, dict):
            return result
        record = dict(result)
        record["sessionId"] = request.sessionId
        record["step"] = request.step
        evaluation_output = result.get("evaluation_output")
        if isinstance(evaluation_output, dict):
            record["demonstratedBloomLevel"] = evaluation_output.get(
                "demonstrated_bloom_level"
            )
        if request.objectiveId is not None:
            record["objectiveId"] = request.objectiveId
        return record

    def _run_step(self, request: EvaluationRequest) -> Dict[str, Any]:
        if request.step > 1:
            # Session already hydrated by handle(); agent/LLM failures propagate
            # → classified retryable by defaults.
            return self.agent.handle_user_answer(request.session_id, request.student_answer)

        # step == 1: create the session from context_id, then process the
        # first answer against the generated first question.
        context = self._context_for(request)
        self.agent.start_session(
            session_id=request.session_id,
            task_title=context.get("task_title", ""),
            task_description=context.get("task_description", ""),
            task_details=context.get("task_details", ""),
        )
        return self.agent.handle_user_answer(request.session_id, request.student_answer)

    def _context_for(self, request: EvaluationRequest) -> Dict[str, Any]:
        if self.context_resolver is None:
            # Backend context resolution (context_id -> task context) lands with
            # the backend wiring; provide a minimal default so a session can start.
            return {
                "task_title": request.context_id,
                "task_description": "",
                "task_details": request.context_id,
            }
        return self.context_resolver(request.context_id)


if __name__ == "__main__":  # pragma: no cover - manual run entrypoint
    import logging

    logging.basicConfig(level=logging.INFO)
    EvaluatorWorker().run()
