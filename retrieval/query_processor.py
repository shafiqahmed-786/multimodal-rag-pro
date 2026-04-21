"""
Query processor — rewrites and expands queries via LLM (Google Gemini).

Upgrades:
  - Exponential backoff retry (Fix 3b)
  - Robust JSON extraction — handles preamble, markdown fences (Fix 3)
  - In-memory rewrite cache with TTL (Fix 4d)
  - Sub-query count capped by QueryPlanner config
  - Year-aware query enrichment injects tabular sub-queries (Fix 2c)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Optional

from google import genai
from google.genai import types

from core.interfaces import BaseQueryProcessor
from core.llm_utils import extract_json, llm_call_with_retry
from core.models import QueryExpansion
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")

_REWRITE_SYSTEM = """You are a search query optimiser for a document retrieval system.
Your job is to:
1. Rewrite the user's query to be more precise and retrieval-friendly
2. Generate 2-3 alternative sub-queries that cover different aspects

Respond ONLY with valid JSON in this exact format:
{
  "rewritten": "the improved query",
  "sub_queries": ["sub-query 1", "sub-query 2"]
}"""


class _RewriteCache:
    """Simple in-memory TTL cache for query rewrites."""

    def __init__(self, ttl: int = 7200) -> None:
        self._store: dict[str, tuple[QueryExpansion, float]] = {}
        self._ttl = ttl

    def get(self, key: str) -> Optional[QueryExpansion]:
        entry = self._store.get(key)
        if entry and time.monotonic() - entry[1] < self._ttl:
            return entry[0]
        return None

    def set(self, key: str, value: QueryExpansion) -> None:
        self._store[key] = (value, time.monotonic())


_rewrite_cache = _RewriteCache(ttl=7200)


def _enrich_with_year_queries(query: str, sub_queries: list[str]) -> list[str]:
    """If query mentions a year, inject a table-targeted sub-query."""
    years = _YEAR_RE.findall(query)
    enriched = list(sub_queries)
    for year in years[:2]:
        tabular = f"table data statistics {year}"
        if tabular not in enriched:
            enriched.append(tabular)
    return enriched


class LLMQueryProcessor(BaseQueryProcessor):
    """
    Uses Gemini to rewrite queries and generate sub-queries for
    multi-query retrieval, with retry and robust JSON parsing.
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def process(self, query: str, max_sub_queries: int = 2) -> QueryExpansion:
        if not settings.enable_query_rewriting:
            return QueryExpansion(original=query, rewritten=query)

        # Cache lookup
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cached = _rewrite_cache.get(cache_key)
        if cached:
            logger.debug("Rewrite cache hit for query: %s", query[:60])
            return cached

        def call() -> str:
            resp = self._client.models.generate_content(
                model=settings.llm_model,
                contents=f"{_REWRITE_SYSTEM}\n\nQuery: {query}",
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=256,
                ),
            )
            return resp.text or "{}"

        def fallback() -> str:
            return json.dumps({"rewritten": query, "sub_queries": []})

        try:
            raw = llm_call_with_retry(call, max_retries=3, fallback_fn=fallback)
            parsed = extract_json(raw)
            sub_queries = parsed.get("sub_queries", [])[:max_sub_queries]
            sub_queries = _enrich_with_year_queries(query, sub_queries)

            result = QueryExpansion(
                original=query,
                rewritten=parsed.get("rewritten", query),
                sub_queries=sub_queries,
            )
        except Exception as exc:
            logger.warning("Query processing failed: %s — using original.", exc)
            result = QueryExpansion(original=query, rewritten=query)

        _rewrite_cache.set(cache_key, result)
        return result
