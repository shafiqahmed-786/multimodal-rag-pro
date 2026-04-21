"""
Query Planner — Upgrade 2.

Classifies incoming query intent and returns a pipeline configuration dict,
allowing the RAGService to skip expensive steps on simple queries.

Pipeline configs:
  SIMPLE  — no rewrite, no rerank, no verify, top_k=3
  FACTUAL — no rewrite, rerank, no verify, top_k=5, table_boost=True
  COMPLEX — full pipeline, top_k=7
"""
from __future__ import annotations

import re

# Signals that indicate a data/numeric lookup — benefit from table boosting
_FACTUAL_SIGNALS = re.compile(
    r"\b(how much|how many|what is the|what was the|percentage|rate|gdp|revenue|"
    r"total|average|growth|figure|number|statistic|data|table|year|"
    r"amount|value|price|cost|count|billion|million|trillion|\d{4})\b",
    re.IGNORECASE,
)

# Signals for complex multi-part analytical queries
_COMPLEX_SIGNALS = re.compile(
    r"\b(compare|why|how does|analyse|analyze|implications|"
    r"difference between|relationship|trend|forecast|impact|evaluate)\b",
    re.IGNORECASE,
)

# Simple definitional / navigational queries
_SIMPLE_SIGNALS = re.compile(
    r"^(what is|who is|define|explain|describe|what does|list)\b",
    re.IGNORECASE,
)


class QueryPlanner:
    """
    Decides which pipeline steps to run for a given query.

    Returns a config dict consumed by RAGService.query():
      {
        "rewrite":     bool,   # run LLM query rewriting
        "rerank":      bool,   # run cross-encoder reranking
        "verify":      bool,   # run hallucination verification (always async)
        "top_k":       int,    # retrieval pool size
        "table_boost": bool,   # boost table chunk scores
        "max_sub_queries": int # cap on LLM-generated sub-queries
      }
    """

    SIMPLE: dict = {
        "rewrite": False,
        "rerank": False,
        "verify": True,
        "top_k": 3,
        "table_boost": False,
        "max_sub_queries": 0,
    }

    FACTUAL: dict = {
        "rewrite": False,
        "rerank": True,
        "verify": True,
        "top_k": 5,
        "table_boost": True,
        "max_sub_queries": 1,
    }

    COMPLEX: dict = {
        "rewrite": True,
        "rerank": True,
        "verify": True,
        "top_k": 7,
        "table_boost": False,
        "max_sub_queries": 2,
    }

    def plan(self, query: str) -> dict:
        """Return pipeline config for the given query."""
        q_lower = query.lower().strip()

        # Factual / data lookup queries
        if _FACTUAL_SIGNALS.search(q_lower):
            plan = dict(self.FACTUAL)
            # If it's also short and simple, reduce top_k
            if len(query) < 60 and not _COMPLEX_SIGNALS.search(q_lower):
                plan["top_k"] = 5
            return plan

        # Complex analytical queries
        if _COMPLEX_SIGNALS.search(q_lower):
            return dict(self.COMPLEX)

        # Simple definitional — short query starting with known pattern
        if _SIMPLE_SIGNALS.match(q_lower) and len(query) < 80:
            return dict(self.SIMPLE)

        # Default: factual (safe baseline)
        return dict(self.FACTUAL)
