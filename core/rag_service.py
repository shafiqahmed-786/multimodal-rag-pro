"""
RAGService — the central orchestrator that wires all components together.

This is the single entry point for both the FastAPI backend and the
Streamlit UI. It owns the full query pipeline:

  query
    → QueryPlanner (adaptive pipeline config)
    → query rewriting + expansion (conditional + cached)
    → multi-query parallel retrieval (ThreadPoolExecutor)
    → StructuredReasoningLayer (no-LLM path for table data)
    → cross-encoder reranking (conditional)
    → answer generation (sync or streaming)
    → hallucination verification (async background — does not block response)
    → citation grounding
    → latency / debug metadata

Key improvements:
  - Parallel sub-query retrieval (Fix 4a)
  - Conditional reranking — only when retrieval pool > top_k (Fix 4b)
  - Async background verification — never blocks the user response (Fix 4c)
  - Query planner — skips expensive steps for simple queries (Upgrade 2)
  - Structured reasoning — no-LLM table lookup (Upgrade 3)
  - MMR-based retrieval diversity (via HybridRetriever)
  - Rewrite cache integration (Fix 4d)
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncIterator, Optional

from config.settings import get_settings
from core.models import (
    Answer,
    AnswerConfidence,
    QueryExpansion,
    QueryRequest,
    RetrievedChunk,
    VerificationResult,
)
from chunking.semantic_chunker import SemanticChunker
from embeddings.dense_encoder import DenseEncoder
from generation.answer_generator import AnswerGenerator
from generation.verifier import AnswerVerifier
from ingestion.pipeline import IngestionPipeline
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_planner import QueryPlanner
from retrieval.query_processor import LLMQueryProcessor
from retrieval.reranker import CrossEncoderReranker
from retrieval.structured_reasoning import StructuredReasoningLayer
from cache.query_cache import QueryCache

logger = logging.getLogger(__name__)
settings = get_settings()

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_RETRIEVAL_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval")


class RAGService:
    """
    Singleton-friendly service that encapsulates the full RAG pipeline.
    Instantiate once; reuse across requests.
    """

    def __init__(self) -> None:
        self._encoder = DenseEncoder()
        self._retriever = HybridRetriever(encoder=self._encoder)
        self._chunker = SemanticChunker()
        self._ingestion = IngestionPipeline(
            chunker=self._chunker,
            retriever=self._retriever,
        )
        self._query_processor = LLMQueryProcessor()
        self._reranker = CrossEncoderReranker()
        self._generator = AnswerGenerator()
        self._verifier = AnswerVerifier()
        self._planner = QueryPlanner()
        self._structured = StructuredReasoningLayer()
        self._cache = QueryCache(
            max_size=settings.query_cache_max_size,
            ttl=settings.cache_ttl,
        )
        self._is_ready = False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, file_path: str, progress_callback=None) -> dict:
        chunks = self._ingestion.ingest_file(file_path, progress_callback)
        self._is_ready = True
        return {"chunks_indexed": len(chunks), "file": Path(file_path).name}

    def save_index(self, path: str | None = None) -> None:
        dest = path or settings.index_dir
        self._retriever.save(dest)
        logger.info("Index saved to %s", dest)

    def load_index(self, path: str | None = None) -> None:
        src = path or settings.index_dir
        self._retriever.load(src)
        self._is_ready = self._retriever.chunk_count > 0

    # ------------------------------------------------------------------
    # Query (sync)
    # ------------------------------------------------------------------

    def query(self, request: QueryRequest) -> Answer:
        t0 = time.perf_counter()

        # --- Cache hit ---
        cache_key = f"{request.text}:k={request.top_k}:mode={request.retrieval_mode}"
        if settings.cache_enabled:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for query: %s", request.text[:60])
                return cached

        # --- Step 0: Query Planner (adaptive pipeline config) ---
        plan = self._planner.plan(request.text)

        # Request flags override planner for explicit user choices
        use_rewrite = request.enable_query_rewriting and plan["rewrite"]
        use_rerank = request.enable_reranking and plan["rerank"]
        top_k = request.top_k
        max_sub_queries = plan.get("max_sub_queries", 2)

        # --- Step 1: Query rewriting ---
        expansion: QueryExpansion | None = None
        effective_query = request.text
        if use_rewrite:
            expansion = self._query_processor.process(
                request.text, max_sub_queries=max_sub_queries
            )
            effective_query = expansion.rewritten
            logger.debug("Query rewritten: %s → %s", request.text, effective_query)

        # --- Step 2: Parallel multi-query retrieval ---
        all_chunks = self._retrieve_multi_parallel(effective_query, expansion, top_k)

        # --- Step 3: Structured reasoning (no-LLM table lookup) ---
        direct_answer = self._structured.try_direct_answer(request.text, all_chunks)

        # --- Step 4: Conditional reranking ---
        final_chunks = all_chunks
        if use_rerank and len(all_chunks) > top_k:
            final_chunks = self._reranker.rerank(effective_query, all_chunks, top_k=top_k)

        final_chunks = final_chunks[:top_k]

        # --- Step 5: Answer generation (skip if direct answer available) ---
        if direct_answer:
            answer_text = direct_answer
            logger.info("Structured reasoning answered query — LLM generation skipped.")
        else:
            answer_text = self._generator.generate(effective_query, final_chunks)

        # --- Step 6: Build citations ---
        citations = self._generator.build_citations(final_chunks)

        latency_ms = (time.perf_counter() - t0) * 1000

        # --- Step 7: Build answer (verification runs async — does not block) ---
        answer = Answer(
            query_id=request.id,
            text=answer_text,
            citations=citations,
            retrieved_chunks=final_chunks,
            query_expansion=expansion,
            verification=VerificationResult(
                is_faithful=True,
                confidence=AnswerConfidence.UNVERIFIED,
                verification_note="Verification running in background…",
            ),
            confidence=AnswerConfidence.UNVERIFIED,
            latency_ms=latency_ms,
            retrieval_debug={
                "num_retrieved": len(all_chunks),
                "num_after_rerank": len(final_chunks),
                "effective_query": effective_query,
                "sub_queries": expansion.sub_queries if expansion else [],
                "plan": plan,
                "direct_answer_used": bool(direct_answer),
            },
        )

        # Cache immediately so subsequent identical queries return fast
        if settings.cache_enabled:
            self._cache.set(cache_key, answer)

        # Fire-and-forget background verification (updates cache entry)
        self._run_verification_background(
            answer, effective_query, answer_text, final_chunks, cache_key
        )

        logger.info(
            "Query answered in %.0f ms | chunks=%d | plan=%s | direct=%s",
            latency_ms, len(final_chunks),
            "complex" if plan["rewrite"] else "factual",
            bool(direct_answer),
        )
        return answer

    # ------------------------------------------------------------------
    # Query (async streaming)
    # ------------------------------------------------------------------

    async def query_stream(self, request: QueryRequest) -> AsyncIterator[str]:
        plan = self._planner.plan(request.text)
        expansion: QueryExpansion | None = None
        effective_query = request.text

        if request.enable_query_rewriting and plan["rewrite"]:
            expansion = self._query_processor.process(
                request.text, max_sub_queries=plan.get("max_sub_queries", 2)
            )
            effective_query = expansion.rewritten

        all_chunks = self._retrieve_multi_parallel(effective_query, expansion, request.top_k)

        final_chunks = all_chunks
        if request.enable_reranking and plan["rerank"] and len(all_chunks) > request.top_k:
            final_chunks = self._reranker.rerank(
                effective_query, all_chunks, top_k=request.top_k
            )
        final_chunks = final_chunks[:request.top_k]

        async for token in self._generator.stream(effective_query, final_chunks):
            yield token

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve_multi_parallel(
        self,
        primary_query: str,
        expansion: QueryExpansion | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Retrieve for primary + sub-queries in parallel threads, then merge."""
        queries = [primary_query]
        if expansion and expansion.sub_queries:
            queries.extend(expansion.sub_queries[:2])   # cap at 2

        def retrieve_one(q: str) -> list[RetrievedChunk]:
            return self._retriever.retrieve(q, top_k=top_k * 2)

        # Parallel retrieval via thread pool
        futures = [_RETRIEVAL_EXECUTOR.submit(retrieve_one, q) for q in queries]
        all_results = [f.result() for f in futures]

        # Merge with content-hash dedup
        seen_ids: set[str] = set()
        merged: list[RetrievedChunk] = []
        for results in all_results:
            for rc in results:
                cid = rc.chunk.id
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    merged.append(rc)

        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[: top_k * 2]

    def _run_verification_background(
        self,
        answer: Answer,
        query: str,
        answer_text: str,
        chunks: list[RetrievedChunk],
        cache_key: str,
    ) -> None:
        """
        Run verification in a background thread.
        Updates the cached answer when done — the next identical query
        will return the verified version.
        """
        def _verify():
            try:
                result = self._verifier.verify(query, answer_text, chunks)
                answer.verification = result
                answer.confidence = result.confidence
                # Update cache with verified answer
                if settings.cache_enabled:
                    self._cache.set(cache_key, answer)
                logger.debug("Background verification complete: %s", result.confidence)
            except Exception as exc:
                logger.warning("Background verification failed: %s", exc)

        _RETRIEVAL_EXECUTOR.submit(_verify)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def chunk_count(self) -> int:
        return self._retriever.chunk_count
