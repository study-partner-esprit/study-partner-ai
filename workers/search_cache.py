"""Search result cache (F05 / SEARCH-06).

Repeated queries hit the crawler + LLM, which is slow and rate-limited (Apify
rate limits, crawl latency). Caching the validated result keyed by
``userId:hash(query)`` avoids redundant crawls within the TTL window. The store
degrades gracefully: when Redis is unreachable or not configured, an in-memory
store still provides per-process caching; a hard failure never fails the job —
the caller treats a cache miss as "compute fresh".
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

SEARCH_CACHE_TTL_SECONDS = 60 * 60  # SEARCH-06: TTL 1h


class SearchCache:
    async def get(self, key: str) -> Optional[str]:
        """Return cached JSON result or None on miss/error."""
        ...  # pragma: no cover

    async def put(self, key: str, value: str) -> None:
        """Store JSON result keyed by *key* with the 1h TTL."""
        ...  # pragma: no cover


class InMemorySearchCache(SearchCache):
    """Process-local fallback (tests / no Redis)."""

    def __init__(self, ttl_seconds: int = SEARCH_CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._data: dict[str, tuple[float, str]] = {}

    async def get(self, key: str) -> Optional[str]:
        entry = self._data.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        import time

        if time.monotonic() - stored_at >= self._ttl:
            self._data.pop(key, None)
            return None
        return value

    async def put(self, key: str, value: str) -> None:
        import time

        self._data[key] = (time.monotonic(), value)


class RedisSearchCache(SearchCache):
    """SET with 1h TTL — safe across replicas."""

    def __init__(
        self,
        redis_client,
        ttl_seconds: int = SEARCH_CACHE_TTL_SECONDS,
        prefix: str = "ai:search:",
    ) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._prefix = prefix

    async def get(self, key: str) -> Optional[str]:
        try:
            return await self._redis.get(f"{self._prefix}{key}")
        except Exception:  # pragma: no cover - redis unreachable
            return None

    async def put(self, key: str, value: str) -> None:
        try:
            await self._redis.set(f"{self._prefix}{key}", value, ex=self._ttl)
        except Exception:  # pragma: no cover - redis unreachable
            return None


def build_default_search_cache():
    """Redis-backed when REDIS_URL is configured, else in-memory."""
    url = os.getenv("REDIS_URL")
    if not url:
        return InMemorySearchCache()
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(url, decode_responses=True)
        return RedisSearchCache(client)
    except Exception:  # pragma: no cover - redis lib missing/unreachable
        return InMemorySearchCache()


def cache_key(user_id: str, query: str, max_results: int) -> str:
    """Stable key for a query within a user; content-addressed so casing or
    whitespace changes produce a distinct (uncached) query."""
    normalized = " ".join(query.split()).lower()
    digest = hashlib.sha256(f"{user_id}|{normalized}|{max_results}".encode()).hexdigest()
    return digest
