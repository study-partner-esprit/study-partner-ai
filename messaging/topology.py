"""RabbitMQ topology for AI jobs (F01 / AI-COM-06).

Python mirror of `study-partner-api/shared/ai-messaging/topology.js`.
Names and semantics MUST stay identical on both sides — enforced by the
topology-parity tests.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

EXCHANGE_JOBS = "ai.jobs"
EXCHANGE_DELAY = "ai.delay"
EXCHANGE_DLX = "ai.dlx"
EXCHANGE_RESULTS = "ai.results"

RESULT_QUEUE = "ai.results.inbox"

def _retry_delays() -> List[int]:
    """Env override exists for integration tests (tiny delays); production
    uses the canonical 1s → 4s → 16s ladder."""
    import json as _json

    raw = os.getenv("AI_RETRY_DELAYS_MS")
    if not raw:
        return [1000, 4000, 16000]
    parsed = _json.loads(raw)
    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(isinstance(n, int) and n > 0 for n in parsed)
    ):
        raise ValueError("AI_RETRY_DELAYS_MS must be a JSON array of positive integers")
    return parsed


RETRY_DELAYS_MS: List[int] = _retry_delays()
MAX_RETRIES = len(RETRY_DELAYS_MS)

RETRY_HEADER = "x-retry-count"

DEFAULT_RABBITMQ_URL = "amqp://guest:guest@localhost:5672/%2F"


def rabbitmq_url() -> str:
    return os.getenv("RABBITMQ_URL", DEFAULT_RABBITMQ_URL)


def work_queue_name(job_type: str) -> str:
    return f"ai.work.{job_type}"


def dlq_queue_name(job_type: str) -> str:
    return f"ai.dlq.{job_type}"


def delay_queue_name(job_type: str, delay_ms: int) -> str:
    return f"ai.delay.{job_type}.{delay_ms}"


def retry_routing_key(job_type: str, delay_ms: int) -> str:
    """Routing key under which a delayed retry is published.  The same key is
    bound from the jobs exchange to the work queue so expired messages land
    back on the correct work queue."""
    return f"retry.{job_type}.{delay_ms}"


def work_queue_arguments(job_type: str) -> Dict[str, Any]:
    """Work queues dead-letter terminal failures to ai.dlx. The routing key is
    pinned to the bare type because a retried message's CURRENT key is
    `retry.<type>.<ms>` — without the override the DLQ binding would miss it."""
    return {
        "x-dead-letter-exchange": EXCHANGE_DLX,
        "x-dead-letter-routing-key": job_type,
    }


def delay_queue_arguments(delay_ms: int) -> Dict[str, Any]:
    """Delay queues: message expires → dead-letters to ai.jobs with the
    CURRENT routing key (= retry.<type>.<ms>), which is bound from the
    jobs exchange to the work queue."""
    return {
        "x-message-ttl": delay_ms,
        "x-dead-letter-exchange": EXCHANGE_JOBS,
    }


async def declare_topology(channel: Any) -> None:
    """Idempotently declare exchanges, work queue/DLQ pairs and delay queues.

    `channel` is an aio-pika channel. Safe to call from every worker at
    startup; declaration arguments must match across processes.
    """
    await channel.declare_exchange(EXCHANGE_JOBS, type="direct", durable=True)
    await channel.declare_exchange(EXCHANGE_DELAY, type="direct", durable=True)
    await channel.declare_exchange(EXCHANGE_DLX, type="direct", durable=True)
    await channel.declare_exchange(EXCHANGE_RESULTS, type="direct", durable=True)
