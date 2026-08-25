"""BaseAIWorker — shared RabbitMQ consumer framework (F01 / AI-COM-05).

All five AI agents (Planner, Coach, Evaluator, Search, Ingestion) inherit this
class instead of running five independent consumers. It owns:

- connection + channel lifecycle with graceful SIGTERM drain
- topology declaration (exchanges, work queue, DLQ, delay queues)
- envelope validation against the AI-COM-02 contract before dispatch
- idempotent dispatch by messageId (AI-COM-08)
- retry with exponential backoff via TTL'd delay queues, terminal failures
  dead-lettered immediately (AI-COM-06)
- validated result events on ai.results (AI-COM-03)

Subclasses implement `handle(payload, envelope) -> dict`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from typing import Any, Dict, Optional

from messaging.envelope import AiJobEnvelope, validate_job_envelope
from messaging.failures import (
    FailureClass,
    RetryableError,
    TerminalError,
    classify_failure,
    sanitized_error,
)
from messaging.topology import (
    EXCHANGE_DELAY,
    EXCHANGE_DLX,
    EXCHANGE_JOBS,
    EXCHANGE_RESULTS,
    MAX_RETRIES,
    RETRY_DELAYS_MS,
    RETRY_HEADER,
    declare_topology,
    dlq_queue_name,
    delay_queue_name,
    delay_queue_arguments,
    retry_routing_key,
    work_queue_arguments,
    work_queue_name,
)
from workers.idempotency import build_default_store

_DEFAULT_RABBITMQ_URL = None


def _default_url() -> str:
    global _DEFAULT_RABBITMQ_URL
    if _DEFAULT_RABBITMQ_URL is None:
        from messaging.topology import rabbitmq_url as _url

        _DEFAULT_RABBITMQ_URL = _url
    return _DEFAULT_RABBITMQ_URL()

logger = logging.getLogger(__name__)


class BaseAIWorker:
    """Consume `job_type` messages; subclass and implement handle()."""

    job_type: str = ""

    def __init__(
        self,
        rabbitmq_url: Optional[str] = None,
        idempotency_store=None,
        prefetch: int = 1,
    ) -> None:
        if not self.job_type:
            raise ValueError("subclasses must define job_type")
        self._url = rabbitmq_url or _default_url()
        self.store = idempotency_store or build_default_store()
        self.prefetch = prefetch

        self._connection = None  # aio_pika RobustConnection
        self._channel = None
        self._queue = None
        self._consumer_tag: Optional[str] = None
        self._shutdown = asyncio.Event()
        self._in_flight: asyncio.Task | None = None

    # ------------------------------------------------------------------ API

    async def handle(self, payload: Dict[str, Any], envelope: AiJobEnvelope) -> Dict[str, Any]:
        """Process one validated job payload; return the result payload."""
        raise NotImplementedError

    async def start(self) -> None:
        import aio_pika

        self._connection = await aio_pika.connect_robust(self._url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=self.prefetch)
        await declare_topology(self._channel)

        queue = await self._channel.declare_queue(
            work_queue_name(self.job_type),
            durable=True,
            arguments=work_queue_arguments(),
        )
        await queue.bind(EXCHANGE_JOBS, routing_key=self.job_type)

        dlq = await self._channel.declare_queue(dlq_queue_name(self.job_type), durable=True)
        await dlq.bind(EXCHANGE_DLX, routing_key=self.job_type)

        for delay in RETRY_DELAYS_MS:
            qname = delay_queue_name(self.job_type, delay)
            dq = await self._channel.declare_queue(
                qname, durable=True, arguments=delay_queue_arguments(delay)
            )
            rk = retry_routing_key(self.job_type, delay)
            await dq.bind(EXCHANGE_DELAY, routing_key=rk)
            # Extra binding: expired messages keep routing_key = retry key,
            # so the work queue must accept it.
            await queue.bind(EXCHANGE_JOBS, routing_key=rk)

        self._consumer_tag = await queue.consume(self.on_message)
        logger.info("worker_started", extra={"job_type": self.job_type})

    async def stop(self) -> None:
        """Graceful drain: stop consuming, let the in-flight handler finish."""
        self._shutdown.set()
        try:
            if self._queue is not None and self._consumer_tag:
                await self._queue.cancel(self._consumer_tag)
            if self._in_flight is not None:
                await asyncio.wait([self._in_flight], timeout=30)
        finally:
            if self._connection is not None:
                await self._connection.close()

    def run(self) -> None:
        """Blocking entrypoint with SIGTERM/SIGINT handling."""

        async def _run() -> None:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            await self.start()
            await self._shutdown.wait()

        asyncio.run(_run())

    # -------------------------------------------------------------- consume

    async def on_message(self, message) -> None:  # aio_pika.IncomingMessage
        """Full ACK/NACK pipeline. Never raises."""
        async with message.process(ignore_processed=True):
            try:
                envelope = validate_job_envelope(json.loads(message.body))
            except Exception as exc:
                logger.warning("invalid_envelope_dead_lettered", extra={"error": str(exc)})
                raise TerminalError(f"invalid envelope: {exc}") from exc

            if not await self.store.claim(envelope.messageId):
                logger.info("duplicate_message_acknowledged", extra={"messageId": envelope.messageId})
                return  # processed inside message.process context → ACKed

            self._in_flight = asyncio.current_task()
            try:
                result_payload = await self.handle(envelope.payload, envelope)
            except _RetryScheduled:
                return  # original ACKed; the delayed copy continues the lifecycle
            except Exception as exc:
                try:
                    await self._on_failure(message, envelope, exc)
                except _RetryScheduled:
                    return  # swallowed; acked inside _on_failure
                return
            finally:
                self._in_flight = None

            await self._publish_result(
                envelope,
                status="completed",
                payload=result_payload or {},
            )

    # ------------------------------------------------------------ internals

    def _attempt(self, message) -> int:
        headers = message.headers or {}
        return int(headers.get(RETRY_HEADER, 0))

    async def _on_failure(self, message, envelope: AiJobEnvelope, exc: Exception) -> None:
        attempt = self._attempt(message)
        classification = classify_failure(exc)
        if isinstance(exc, TerminalError):
            classification = FailureClass.TERMINAL
        elif isinstance(exc, RetryableError):
            classification = FailureClass.RETRYABLE

        reason = sanitized_error(exc)
        if attempt >= MAX_RETRIES or classification == FailureClass.TERMINAL:
            await self._publish_result(envelope, status="failed", error=reason)
            logger.error("job_dead_lettered", extra={
                "messageId": envelope.messageId,
                "correlationId": envelope.correlationId,
                "reason": reason,
                "attempts": attempt + 1,
            })
            raise TerminalError(reason) from exc  # DLX routes to per-type DLQ

        delay_ms = RETRY_DELAYS_MS[attempt]
        await self._republish_for_retry(message, envelope, attempt + 1, delay_ms, reason)
        logger.warning("job_scheduled_for_retry", extra={
            "messageId": envelope.messageId,
            "delayMs": delay_ms,
            "nextAttempt": attempt + 1,
        })
        # ACK original: its copy already waits on a delay queue.

    async def _republish_for_retry(
        self, message, envelope: AiJobEnvelope, next_attempt: int, delay_ms: int, reason: str
    ) -> None:
        import aio_pika

        headers = dict(message.headers or {})
        headers[RETRY_HEADER] = next_attempt
        headers["x-last-failure"] = reason[:256]
        delay_exchange = await self._channel.get_exchange(EXCHANGE_DELAY)
        rk = retry_routing_key(envelope.type, delay_ms)
        await delay_exchange.publish(
            aio_pika.Message(
                body=message.body,
                headers=headers,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json",
                message_id=envelope.messageId,
                correlation_id=envelope.correlationId,
            ),
            routing_key=rk,
        )
        await message.ack()  # original consumed; delayed copy continues
        raise _RetryScheduled()

    async def _publish_result(
        self,
        envelope: AiJobEnvelope,
        *,
        status: str,
        payload: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        import aio_pika

        from messaging.envelope import AiResultEnvelope

        body: Dict[str, Any] = {
            "messageId": envelope.messageId,
            "correlationId": envelope.correlationId,
            "type": envelope.type,
            "version": envelope.version,
            "requestId": envelope.requestId,
            "timestamp": envelope.timestamp.isoformat().replace("+00:00", "Z"),
            "status": status,
        }
        if payload is not None:
            body["payload"] = payload
        if error is not None:
            body["error"] = error

        AiResultEnvelope.model_validate(body)  # never publish an invalid event

        exchange = await self._channel.get_exchange(EXCHANGE_RESULTS)
        await exchange.publish(
            aio_pika.Message(
                body=json.dumps(body).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json",
                message_id=body["messageId"],
                correlation_id=body["correlationId"],
                type="result",
            ),
            routing_key="result",
        )


class _RetryScheduled(Exception):
    """Internal control flow: retry copy published + original ACKed."""
