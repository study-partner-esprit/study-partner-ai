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

RETRY_DELAYS_MS: List[int] = [1000, 4000, 16000]
MAX_RETRIES = len(RETRY_DELAYS_MS)

RETRY_HEADER = "x-retry-count"

DEFAULT_RABBITMQ_URL = "amqp://guest:guest@localhost:5672/%2F"


def rabbitmq_url() -> str:
    return os.getenv("RABBITMQ_URL", DEFAULT_RABBITMQ_URL)


def work_queue_name(job_type: str) -> str:
    return f"ai.work.{job_type}"


def dlq_queue_name(job_type: str) -> str:
    return f"ai.dlq.{job_type}"


def delay_queue_name(delay_ms: int) -> str:
    if delay_ms in RETRY_DELAYS_MS:
        return f"ai.delay.{delay_ms // 1000}s"
    return f"ai.delay.{delay_ms}ms"


def work_queue_arguments() -> Dict[str, Any]:
    """Work queues dead-letter terminal failures to ai.dlx."""
    return {"x-dead-letter-exchange": EXCHANGE_DLX}


def delay_queue_arguments(delay_ms: int) -> Dict[str, Any]:
    """Delay queues expire then re-route to their original work queue:
    dead-lettering preserves the original routing key (= job type), and the
    DLX here is the jobs exchange itself."""
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
