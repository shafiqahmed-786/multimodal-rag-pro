"""
Sparse retrieval via BM25 (rank_bm25).

Returns normalised BM25 scores as a list of floats (not true vectors).
The SparseRetriever uses these scores directly.
"""
from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKENISE_RE = re.compile(r"\W+")


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in _TOKENISE_RE.split(text) if t]


class BM25Index:
    """
    Wraps rank_bm25.BM25Okapi for corpus indexing and scoring.
    """

    def __init__(self) -> None:
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def fit(self, texts: list[str]) -> None:
        self._corpus_tokens = [_tokenise(t) for t in texts]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info("BM25 index built with %d documents.", len(texts))

    def get_scores(self, query: str) -> list[float]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built — call fit() first.")
        query_tokens = _tokenise(query)
        scores = self._bm25.get_scores(query_tokens)
        # Normalise to [0, 1]
        max_score = float(scores.max()) if scores.size > 0 else 1.0
        if max_score > 0:
            scores = scores / max_score
        return scores.tolist()

    def top_k(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Returns [(index, score)] sorted descending."""
        scores = self.get_scores(query)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return indexed[:k]

    def __len__(self) -> int:
        return len(self._corpus_tokens)
