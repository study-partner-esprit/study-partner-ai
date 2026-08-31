"""Session state store tests (F04 / EVAL-02 rehydration)."""

from __future__ import annotations

import asyncio

import pytest

from workers.session_store import InMemorySessionStore


async def test_put_get_roundtrip():
    store = InMemorySessionStore()
    await store.put("sess-1", '{"session_id": "sess-1"}')
    assert await store.get("sess-1") == '{"session_id": "sess-1"}'


async def test_get_missing_returns_none():
    store = InMemorySessionStore()
    assert await store.get("does-not-exist") is None


async def test_put_replaces_and_delete_removes():
    store = InMemorySessionStore()
    await store.put("sess-1", "old")
    await store.put("sess-1", "new")
    assert await store.get("sess-1") == "new"

    await store.delete("sess-1")
    assert await store.get("sess-1") is None


async def test_get_after_ttl_expiry_returns_none():
    store = InMemorySessionStore(ttl_seconds=1)
    await store.put("sess-1", "payload")
    assert await store.get("sess-1") == "payload"
    await asyncio.sleep(1.2)
    assert await store.get("sess-1") is None


async def test_concurrent_workers_share_store_via_async_calls():
    store = InMemorySessionStore()
    await asyncio.gather(*[
        store.put(f"sess-{i}", f'{{"i": {i}}}') for i in range(50)
    ])
    assert await store.get("sess-42") == '{"i": 42}'


def test_build_default_store_returns_in_memory_without_redis():
    from workers.session_store import build_default_session_store

    store = build_default_session_store()
    assert isinstance(store, InMemorySessionStore)
