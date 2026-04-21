"""
Structured Reasoning Layer — Upgrade 3 / Differentiator 2.

For numeric data lookup queries, attempts to answer directly from structured
table data (pandas) WITHOUT an LLM call.

Benefits:
  - Zero hallucination on table values (exact row lookup)
  - Instant response for numeric lookups (~0ms vs 4-6s LLM call)
  - Confidence = 1.0 (exact match from source data)

Integrates between retrieval and generation in RAGService.query().
If try_direct_answer() returns a string, generation is skipped entirely.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from core.models import ContentType, RetrievedChunk

logger = logging.getLogger(__name__)

# Patterns for year and common numeric lookup intents
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")

_NUMERIC_INTENTS = re.compile(
    r"\b(gdp|revenue|growth|rate|percentage|total|amount|value|figure|"
    r"billion|million|trillion|percent|ratio|share|index|score|number|count)\b",
    re.IGNORECASE,
)

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    logger.warning("pandas not installed — structured reasoning layer disabled.")


class StructuredReasoningLayer:
    """
    Attempts programmatic answers from table chunks before falling back to LLM.

    Usage:
        layer = StructuredReasoningLayer()
        direct = layer.try_direct_answer(query, retrieved_chunks)
        if direct:
            return direct   # skip LLM generation entirely
        else:
            answer = llm.generate(query, chunks)
    """

    def try_direct_answer(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> Optional[str]:
        """
        Return a direct answer string if the query can be answered programmatically
        from structured table data. Returns None to fall through to LLM generation.
        """
        if not _PANDAS_AVAILABLE:
            return None

        # Only attempt if query looks like a numeric data lookup
        if not _NUMERIC_INTENTS.search(query):
            return None

        years = _YEAR_RE.findall(query)
        table_chunks = [
            rc for rc in chunks
            if rc.chunk.content_type == ContentType.TABLE
            and rc.chunk.metadata.get("structured_data")
        ]

        if not table_chunks:
            return None

        answers = []

        for rc in table_chunks:
            try:
                result = self._lookup(query, rc, years)
                if result:
                    answers.append(result)
            except Exception as exc:
                logger.debug("Structured lookup failed for chunk %s: %s", rc.chunk.id, exc)

        if not answers:
            return None

        # Return first confident match
        source = answers[0]
        logger.info("Structured reasoning answered query directly (no LLM needed).")
        return source

    def _lookup(
        self, query: str, rc: RetrievedChunk, years: list[str]
    ) -> Optional[str]:
        """Attempt a pandas-based lookup on a single table chunk."""
        import pandas as pd  # local import — already checked above

        data = json.loads(rc.chunk.metadata["structured_data"])
        headers = data.get("headers", [])
        rows = data.get("rows", [])

        if not rows or not headers:
            return None

        df = pd.DataFrame(rows)
        if df.empty:
            return None

        # Identify year column
        year_cols = [c for c in df.columns if re.search(r"year|date|period|fy", c, re.IGNORECASE)]

        if years and year_cols:
            year_col = year_cols[0]
            mask = df[year_col].astype(str).str.contains("|".join(years), na=False)
            matched = df[mask]
            if not matched.empty:
                chunk_ref = f"{rc.chunk.source_file}, page {rc.chunk.page_number}"
                records = matched.to_dict("records")
                rows_text = "; ".join(
                    ", ".join(f"{k}: {v}" for k, v in rec.items()) for rec in records
                )
                return (
                    f"Based on structured table data from {chunk_ref}:\n{rows_text}.\n"
                    f"[SOURCE: table on page {rc.chunk.page_number}]"
                )

        # Fallback: return full table summary if small enough
        if len(df) <= 15:
            chunk_ref = f"{rc.chunk.source_file}, page {rc.chunk.page_number}"
            return (
                f"Relevant table from {chunk_ref} (columns: {', '.join(headers)}):\n"
                + df.to_string(index=False)
                + f"\n[SOURCE: table on page {rc.chunk.page_number}]"
            )

        return None
