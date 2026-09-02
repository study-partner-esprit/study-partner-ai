"""Evaluation session state store (F04 / EVAL-02 rehydration).

Keeps serialized ``EvaluationSession`` objects keyed by ``session_id`` so a
multi-turn Socratic session can resume after a worker restart (the agent's
in-memory ``self.sessions`` dict is not reliable across restarts). Used by
``EvaluatorWorker`` to persist after each step and to rehydrate a session the
agent no longer holds.

In-memory by default (tests / single worker); a Redis-backed variant follows
the same pattern as ``workers/idempotency.py`` and is selected automatically
when ``REDIS_URL`` is configured.
"""

from __future__ import annotations

import time
from typing import Optional, Protocol


class SessionStore(Protocol):
    async def get(self, session_id: str) -> Optional[str]:
        """Return the serialized session JSON, or None if absent/expired."""
        ...  # pragma: no cover

    async def put(self, session_id: str, serialized: str) -> None:
        """Store the serialized session JSON, replacing any prior value."""
        ...  # pragma: no cover

    async def delete(self, session_id: str) -> None:
        """Remove a session (e.g. terminal sessions can be expired)."""
        ...  # pragma: no cover


class InMemorySessionStore:
    """Single-process fallback (tests / single worker deployments)."""

    def __init__(self, ttl_seconds: int = 86400) -> None:
        self._ttl = ttl_seconds
        self._data: dict[str, tuple[float, str]] = {}

    async def get(self, session_id: str) -> Optional[str]:
        item = self._data.get(session_id)
        if item is None:
            return None
        ts, serialized = item
        if time.monotonic() - ts > self._ttl:
            self._data.pop(session_id, None)
            return None
        return serialized

    async def put(self, session_id: str, serialized: str) -> None:
        # opportunistic pruning keeps the dict bounded in long-lived processes
        if len(self._data) > 100_000:
            now = time.monotonic()
            self._data = {k: v for k, v in self._data.items() if now - v[0] < self._ttl}
        self._data[session_id] = (time.monotonic(), serialized)

    async def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class RedisSessionStore:
    """SET/GET with TTL — safe across replicas."""

    def __init__(self, redis_client, ttl_seconds: int = 86400, prefix: str = "ai:evalsess:") -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._prefix = prefix

    async def get(self, session_id: str) -> Optional[str]:
        return await self._redis.get(f"{self._prefix}{session_id}")

    async def put(self, session_id: str, serialized: str) -> None:
        await self._redis.set(
            f"{self._prefix}{session_id}", serialized, ex=self._ttl
        )

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(f"{self._prefix}{session_id}")


def build_default_session_store(ttl_seconds: int = 86400):
    """Redis-backed when REDIS_URL is configured, else in-memory."""
    import os

    url = os.getenv("REDIS_URL")
    if not url:
        return InMemorySessionStore(ttl_seconds)
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(url, decode_responses=True)
        return RedisSessionStore(client, ttl_seconds)
    except Exception:  # pragma: no cover - redis lib missing/unreachable
        return InMemorySessionStore(ttl_seconds)
