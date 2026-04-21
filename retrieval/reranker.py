"""
Cross-encoder reranker for precision boosting.

Uses a cross-encoder (query, passage) scorer that is far more accurate
than bi-encoder cosine similarity for final ranking.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from core.interfaces import BaseReranker
from core.models import RetrievedChunk
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    logger.info("Loading cross-encoder: %s", model_name)
    return CrossEncoder(model_name, max_length=512)


class CrossEncoderReranker(BaseReranker):
    """
    Re-scores (query, chunk) pairs with a cross-encoder.
    Significantly improves precision at the cost of latency.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.reranker_model

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        try:
            model = _load_cross_encoder(self._model_name)
            pairs = [(query, rc.chunk.content) for rc in chunks]
            scores = model.predict(pairs, show_progress_bar=False)

            rescored = [
                RetrievedChunk(
                    chunk=rc.chunk,
                    score=float(score),
                    retrieval_method=f"reranked:{rc.retrieval_method}",
                    rank=0,
                )
                for rc, score in zip(chunks, scores)
            ]
            rescored.sort(key=lambda x: x.score, reverse=True)

            # Re-assign ranks
            for i, rc in enumerate(rescored):
                rc.rank = i

            return rescored[:top_k]

        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker failed, returning original order: %s", exc)
            return chunks[:top_k]
