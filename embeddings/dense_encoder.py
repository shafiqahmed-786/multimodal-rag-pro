"""
Dense encoder backed by SentenceTransformers.

Features:
- Batched encoding for memory efficiency
- Normalised embeddings (cosine similarity ready)
- Thread-safe singleton model loading
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from core.interfaces import BaseEncoder
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


class DenseEncoder(BaseEncoder):
    """
    Produces L2-normalised dense embeddings using SentenceTransformers.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None
        self._dim: int | None = None

    @property
    def _loaded_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = _load_model(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dim is None:
            sample = self._loaded_model.encode(["test"], normalize_embeddings=True)
            self._dim = sample.shape[1]
        return self._dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts in batches. Returns normalised float lists."""
        if not texts:
            return []

        batch_size = settings.embedding_batch_size
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._loaded_model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            all_embeddings.append(embeddings)

        combined = np.vstack(all_embeddings)
        return combined.tolist()
