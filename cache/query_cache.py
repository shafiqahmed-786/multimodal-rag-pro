"""
In-memory query cache backed by an LRU dict with TTL expiry.

For production, swap this for Redis (see storage/redis_cache.py stub).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Optional

from core.interfaces import BaseCacheManager


class QueryCache(BaseCacheManager):
    """
    Thread-safe LRU cache with per-entry TTL.
    """

    def __init__(self, max_size: int = 256, ttl: int = 3600) -> None:
        self._max_size = max_size
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        value, expiry = self._store[key]
        if time.monotonic() > expiry:
            del self._store[key]
            return None
        # LRU: move to end
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._ttl
        expiry = time.monotonic() + effective_ttl
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, expiry)
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)  # evict oldest

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    @property
    def stats(self) -> dict:
        now = time.monotonic()
        live = sum(1 for _, (_, exp) in self._store.items() if now <= exp)
        return {"size": len(self._store), "live_entries": live, "max_size": self._max_size}
