"""Idempotency by messageId (F01 / AI-COM-08).

Consumers claim a messageId before processing; a duplicate claim means the
message was already handled and must be ACKed without re-running the handler.
Keys are stored with a TTL (24h default) to bound store growth.
"""

from __future__ import annotations

import time
from typing import Protocol


class IdempotencyStore(Protocol):
    async def claim(self, message_id: str) -> bool:
        """True if claimed (first delivery); False if already processed."""
        ...  # pragma: no cover


class InMemoryIdempotencyStore:
    """Single-process fallback (tests / single worker deployments)."""

    def __init__(self, ttl_seconds: int = 86400) -> None:
        self._ttl = ttl_seconds
        self._seen: dict[str, float] = {}

    async def claim(self, message_id: str) -> bool:
        now = time.monotonic()
        # opportunistic pruning keeps the dict bounded in long-lived processes
        if len(self._seen) > 10_000:
            self._seen = {k: t for k, t in self._seen.items() if now - t < self._ttl}
        expiry = self._seen.get(message_id)
        if expiry is not None and now - expiry < self._ttl:
            return False
        self._seen[message_id] = now
        return True


class RedisIdempotencyStore:
    """SETNX + TTL — safe across replicas."""

    def __init__(self, redis_client, ttl_seconds: int = 86400, prefix: str = "ai:idem:") -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._prefix = prefix

    async def claim(self, message_id: str) -> bool:
        key = f"{self._prefix}{message_id}"
        was_set = await self._redis.set(key, "1", nx=True, ex=self._ttl)
        return bool(was_set)


def build_default_store(ttl_seconds: int = 86400):
    """Redis-backed when REDIS_URL is configured, else in-memory."""
    import os

    url = os.getenv("REDIS_URL")
    if not url:
        return InMemoryIdempotencyStore(ttl_seconds)
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(url, decode_responses=True)
        return RedisIdempotencyStore(client, ttl_seconds)
    except Exception:  # pragma: no cover - redis lib missing/unreachable
        return InMemoryIdempotencyStore(ttl_seconds)
