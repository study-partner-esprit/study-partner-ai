"""SearchWorker — consumes `study.search.query` jobs (F05 / SEARCH-01).

Turns the previously synchronous Flask search proxy into a reliable, async
job-bus consumer. The worker:

- validates the payload into a strict `SearchRequest` (SEARCH-02) — malformed
  payloads are TERMINAL (retrying cannot fix a bad request)
- serves a Redis/in-memory cached result for repeated queries (SEARCH-06,
  TTL 1h) keyed by ``userId:hash(query)`` — skipping the slow crawler+LLM path
- otherwise runs the retrieval → allowlist/SSRF → extraction → prompt-isolation
  → LLM synthesis pipeline off the event loop via to_thread (retrieval + LLM
  are blocking/network-bound; must never block the async loop)
- validates the synthesized output (SEARCH-05) into a strict `SearchOutput`
- degrades gracefully: transient crawler/LLM unavailability yields a degraded
  result (never an exception), so a search job completes instead of retrying
  or dead-lettering on environmental flakiness (SEARCH-06)

Identity (`userId`) comes only from the authenticated envelope, never the
payload.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from pydantic import ValidationError

from messaging.failures import TerminalError
from workers.base import BaseAIWorker
from workers.schemas import SearchRequest
from workers.search_cache import cache_key, build_default_search_cache


class SearchWorker(BaseAIWorker):
    job_type = "study.search.query"

    def __init__(
        self,
        rabbitmq_url: Optional[str] = None,
        idempotency_store=None,
        prefetch: int = 1,
        cache=None,
        pipeline_runner=None,
    ) -> None:
        super().__init__(
            rabbitmq_url=rabbitmq_url,
            idempotency_store=idempotency_store,
            prefetch=prefetch,
        )
        # SEARCH-06 result cache; in-memory fallback when Redis is absent.
        self.cache = cache or build_default_search_cache()
        # Injectable pipeline entrypoint (defaults to the real one). Injected in
        # tests to avoid real web crawling + LLM calls.
        self._pipeline_runner = pipeline_runner

    @property
    def pipeline_runner(self):
        if self._pipeline_runner is None:
            from agents.search.pipeline import run_pipeline

            self._pipeline_runner = run_pipeline
        return self._pipeline_runner

    async def handle(self, payload: Dict[str, Any], envelope) -> Dict[str, Any]:
        request = self._build_request(payload)
        user_id = envelope.userId

        # SEARCH-06: serve cached result keyed by user + query when present.
        key = cache_key(user_id, request.query, request.max_results)
        cached = await self.cache.get(key)
        if cached is not None:
            try:
                cached_obj = json.loads(cached)
            except (ValueError, TypeError):
                cached_obj = None
            if isinstance(cached_obj, dict) and "answer" in cached_obj:
                return self._with_metadata(cached_obj, request, cached=True)

        # Search pipeline is blocking (HTTP crawling + LLM) → off the event loop.
        result = await asyncio.to_thread(
            self._run_pipeline,
            request,
        )

        # SEARCH-05: enforce sources-required + schema conformance.
        result = validate_pipeline_result(result)

        payload_out = {
            "answer": result.answer,
            "sources": list(result.sources),
            "degraded": result.degraded,
            "reason": result.reason,
        }
        if result.voice_summary:
            payload_out["voice_summary"] = result.voice_summary

        # Cache successful (non-degraded) results to avoid repeated crawls.
        if not result.degraded and result.sources:
            await self.cache.put(key, json.dumps(payload_out))

        return self._with_metadata(payload_out, request, cached=False)

    # ------------------------------------------------------------ internals

    def _build_request(self, payload: Any) -> SearchRequest:
        if not isinstance(payload, dict):
            raise TerminalError("invalid search payload: payload must be an object")
        try:
            return SearchRequest.model_validate(payload)
        except ValidationError as e:
            raise TerminalError(f"invalid search payload: {e}") from e

    def _run_pipeline(self, request: SearchRequest) -> Any:
        from agents.search.pipeline import PipelineResult

        try:
            result = self.pipeline_runner(
                request.query,
                max_results=request.max_results,
                use_voice=request.voice_mode,
            )
        except Exception as exc:  # pragma: no cover - defensive; runner degrades
            return PipelineResult(
                answer="Search could not be completed right now. Please try again.",
                sources=[],
                degraded=True,
                reason="search_failed",
            )
        return result

    def _with_metadata(self, payload: Dict[str, Any], request: SearchRequest, *, cached: bool) -> Dict[str, Any]:
        """Add job-level metadata (echo query + flags) to the result payload.

        The Node backend persists the ai.results payload verbatim, so the query
        echo and flags let it store a complete search-history record while the
        cached flag lets tests assert the SEARCH-06 cache path ran.
        """
        out = dict(payload)
        out["query"] = request.query
        out["maxResults"] = request.max_results
        out["voiceMode"] = request.voice_mode
        out["cached"] = cached
        return out


def validate_pipeline_result(result: Any) -> Any:
    """SEARCH-05 validator used by the worker (kept importable for tests).

    Rejects answers with no sources and outputs that fail strict schema
    conformance; falls back to a degraded (valid) result.
    """
    from agents.search.pipeline import validate_pipeline_result as _vp

    return _vp(result)


if __name__ == "__main__":  # pragma: no cover - manual run entrypoint
    import logging

    logging.basicConfig(level=logging.INFO)
    SearchWorker().run()
