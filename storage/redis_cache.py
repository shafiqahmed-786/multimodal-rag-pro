"""
Redis-backed cache — drop-in replacement for QueryCache in production.

Usage:
    Set CACHE_BACKEND=redis in .env.
    Then in RAGService.__init__ swap QueryCache for RedisCache.

Requires: redis>=5.0.0
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from core.interfaces import BaseCacheManager

logger = logging.getLogger(__name__)


class RedisCache(BaseCacheManager):
    """
    Production cache backed by Redis.
    All values are JSON-serialised (non-serialisable objects fall back to
    in-process cache with a warning).
    """

    def __init__(self, url: str = "redis://localhost:6379/0", ttl: int = 3600) -> None:
        try:
            import redis
            self._client = redis.from_url(url, decode_responses=True)
            self._client.ping()
            logger.info("Redis cache connected: %s", url)
        except Exception as exc:
            logger.warning("Redis unavailable (%s) — falling back to no-op cache.", exc)
            self._client = None
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if self._client is None:
            return None
        try:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.debug("Redis GET error: %s", exc)
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if self._client is None:
            return
        try:
            serialised = json.dumps(value, default=str)
            self._client.setex(key, ttl or self._ttl, serialised)
        except Exception as exc:
            logger.debug("Redis SET error: %s", exc)

    def delete(self, key: str) -> None:
        if self._client:
            self._client.delete(key)

    def clear(self) -> None:
        if self._client:
            self._client.flushdb()

    @property
    def stats(self) -> dict:
        if self._client is None:
            return {"status": "disconnected"}
        info = self._client.info("memory")
        return {
            "status": "connected",
            "used_memory_human": info.get("used_memory_human"),
            "db_size": self._client.dbsize(),
        }
