"""
Hybrid retriever: dense (FAISS) + sparse (BM25) fused via Reciprocal Rank Fusion.

Upgrades vs original:
  - Content-hash deduplication (not just ID-based) — prevents duplicate chunks
    from multiple ingestions or sub-query overlaps
  - Index accumulation bug fix: _indexed_sigs set prevents double-indexing
  - Table score boosting: numeric/data queries boost table chunk scores 1.3x
  - MMR (Maximal Marginal Relevance): selects diverse, non-redundant chunks
  - Retrieval explanation: each RetrievedChunk carries dense/sparse/boost scores
  - Year-aware query enrichment: injects tabular sub-queries for year mentions
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from core.interfaces import BaseRetriever
from core.models import Chunk, ContentType, RetrievedChunk
from embeddings.dense_encoder import DenseEncoder
from embeddings.sparse_encoder import BM25Index
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Patterns for adaptive boosting
# ---------------------------------------------------------------------------

_DATA_QUERY_RE = re.compile(
    r"\b(how much|how many|what is the|what was the|percentage|rate|gdp|revenue|"
    r"total|average|growth|figure|number|statistic|data|table|year|"
    r"amount|value|price|cost|billion|million|trillion|\d{4})\b",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")

TABLE_BOOST = 1.3   # score multiplier for table chunks on data queries


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense_results: list[tuple[int, float]],
    sparse_results: list[tuple[int, float]],
    k_rrf: int = 60,
    alpha: float = 0.5,
) -> list[tuple[int, float, float, float]]:
    """
    Fuse two ranked lists using RRF.
    Returns [(global_index, fused_score, dense_score, sparse_score)] sorted desc.
    """
    dense_scores: dict[int, float] = {}
    sparse_scores: dict[int, float] = {}
    fused: dict[int, float] = {}

    for rank, (idx, raw) in enumerate(dense_results):
        contrib = alpha * (1.0 / (k_rrf + rank + 1))
        fused[idx] = fused.get(idx, 0.0) + contrib
        dense_scores[idx] = raw

    for rank, (idx, raw) in enumerate(sparse_results):
        contrib = (1 - alpha) * (1.0 / (k_rrf + rank + 1))
        fused[idx] = fused.get(idx, 0.0) + contrib
        sparse_scores[idx] = raw

    sorted_items = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [
        (idx, score, dense_scores.get(idx, 0.0), sparse_scores.get(idx, 0.0))
        for idx, score in sorted_items
    ]


# ---------------------------------------------------------------------------
# Content hash helper
# ---------------------------------------------------------------------------

def _content_sig(content: str) -> str:
    return hashlib.md5(content[:300].encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    def __init__(self, encoder: DenseEncoder) -> None:
        self.encoder = encoder
        self._index: Optional[faiss.Index] = None
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        texts = [c.content for c in chunks]
        embeddings = np.array(self.encoder.encode(texts), dtype=np.float32)

        if self._index is None:
            self._index = faiss.IndexFlatIP(self.encoder.dimension)

        self._index.add(embeddings)
        self._chunks.extend(chunks)
        logger.debug("Dense index: %d total vectors.", self._index.ntotal)

    def retrieve(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        q_emb = np.array(self.encoder.encode([query]), dtype=np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_emb, k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]


# ---------------------------------------------------------------------------
# Sparse retriever
# ---------------------------------------------------------------------------

class SparseRetriever:
    def __init__(self) -> None:
        self._bm25 = BM25Index()
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        texts = [c.content for c in self._chunks]
        self._bm25.fit(texts)

    def retrieve(self, query: str, top_k: int) -> list[tuple[int, float]]:
        return self._bm25.top_k(query, k=top_k)


# ---------------------------------------------------------------------------
# MMR selection
# ---------------------------------------------------------------------------

def _cosine_sim(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _mmr_select(
    chunks: list[RetrievedChunk],
    top_k: int,
    lambda_mmr: float = 0.6,
) -> list[RetrievedChunk]:
    """
    Maximal Marginal Relevance selection.
    Selects chunks that are both relevant AND diverse (low inter-chunk similarity).
    Only runs if embeddings are available on chunks.
    """
    # Only use MMR if embeddings present
    have_embeddings = all(rc.chunk.embedding for rc in chunks)
    if not have_embeddings or len(chunks) <= top_k:
        return chunks[:top_k]

    selected: list[RetrievedChunk] = []
    candidates = list(chunks)

    while len(selected) < top_k and candidates:
        if not selected:
            best = max(candidates, key=lambda rc: rc.score)
        else:
            sel_embs = [rc.chunk.embedding for rc in selected]

            def mmr_score(rc: RetrievedChunk) -> float:
                relevance = rc.score
                max_sim = max(_cosine_sim(rc.chunk.embedding, e) for e in sel_embs)
                return lambda_mmr * relevance - (1 - lambda_mmr) * max_sim

            best = max(candidates, key=mmr_score)

        selected.append(best)
        candidates.remove(best)

    return selected


# ---------------------------------------------------------------------------
# Hybrid retriever (public interface)
# ---------------------------------------------------------------------------

class HybridRetriever(BaseRetriever):
    """
    Combines FAISS dense + BM25 sparse retrieval with RRF fusion.

    Enhancements:
      - Content-hash dedup prevents duplicate chunks
      - Table boosting improves recall on data-lookup queries
      - MMR diversifies the final result set
      - Retrieval explanation populated per chunk
    """

    def __init__(
        self,
        encoder: Optional[DenseEncoder] = None,
        alpha: Optional[float] = None,
    ) -> None:
        self._encoder = encoder or DenseEncoder()
        self._alpha = alpha if alpha is not None else settings.hybrid_alpha
        self._dense = DenseRetriever(self._encoder)
        self._sparse = SparseRetriever()
        self._chunks: list[Chunk] = []
        self._indexed_sigs: set[str] = set()   # Fix: prevent double-ingestion

    # ------------------------------------------------------------------ index

    def index(self, chunks: list[Chunk]) -> None:
        """Index new chunks, skipping any already indexed by content hash."""
        new_chunks = []
        for chunk in chunks:
            sig = _content_sig(chunk.content)
            if sig not in self._indexed_sigs:
                self._indexed_sigs.add(sig)
                new_chunks.append(chunk)

        if not new_chunks:
            logger.debug("All %d chunks already indexed — skipping.", len(chunks))
            return

        self._dense.index(new_chunks)
        self._sparse.index(new_chunks)
        self._chunks.extend(new_chunks)
        logger.info("Indexed %d new chunks (%d skipped as duplicates).",
                    len(new_chunks), len(chunks) - len(new_chunks))

    # ---------------------------------------------------------------- retrieve

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        dense_raw = self._dense.retrieve(query, top_k=settings.dense_top_k)
        sparse_raw = self._sparse.retrieve(query, top_k=settings.bm25_top_k)

        fused = _rrf_fuse(dense_raw, sparse_raw, alpha=self._alpha)

        # Apply table score boost for data-seeking queries
        is_data_query = bool(_DATA_QUERY_RE.search(query))

        results: list[RetrievedChunk] = []
        seen_sigs: set[str] = set()

        for rank, (idx, score, dense_s, sparse_s) in enumerate(fused):
            if idx >= len(self._chunks):
                continue

            chunk = self._chunks[idx]
            sig = _content_sig(chunk.content)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)

            boost_applied = None
            final_score = score

            if is_data_query and chunk.content_type == ContentType.TABLE:
                final_score *= TABLE_BOOST
                boost_applied = f"table_boost_{TABLE_BOOST}x"

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=final_score,
                    retrieval_method="hybrid",
                    rank=rank,
                    retrieval_explanation={
                        "dense_score": round(dense_s, 4),
                        "sparse_score": round(sparse_s, 4),
                        "rrf_score": round(score, 4),
                        "final_score": round(final_score, 4),
                        "boost_applied": boost_applied,
                        "is_data_query": is_data_query,
                    },
                )
            )

        # Re-sort after potential boost
        results.sort(key=lambda rc: rc.score, reverse=True)

        # MMR diversity selection
        diverse = _mmr_select(results, top_k=top_k * 2)
        return diverse[:top_k * 2]   # caller trims to top_k after reranking

    # --------------------------------------------------------------- persist

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss_path = path / "dense.index"
        meta_path = path / "metadata.pkl"

        if self._dense._index is not None:
            faiss.write_index(self._dense._index, str(faiss_path))

        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "chunks": self._chunks,
                    "alpha": self._alpha,
                    "encoder_model": self._encoder._model_name,
                    "indexed_sigs": self._indexed_sigs,
                },
                f,
            )

        logger.info("Index saved to %s", path)

    def load(self, path: str) -> None:
        path = Path(path)
        faiss_path = path / "dense.index"
        meta_path = path / "metadata.pkl"

        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No saved index at {path}")

        self._dense._index = faiss.read_index(str(faiss_path))

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self._chunks = meta["chunks"]
        self._alpha = meta["alpha"]
        self._dense._chunks = self._chunks
        self._indexed_sigs = meta.get("indexed_sigs", set())

        # Rebuild BM25 from loaded chunks
        self._sparse = SparseRetriever()
        if self._chunks:
            self._sparse.index(self._chunks)

        logger.info(
            "Index loaded: %d chunks, %d FAISS vectors.",
            len(self._chunks),
            self._dense._index.ntotal if self._dense._index else 0,
        )

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
