"""
FastAPI application — production-grade HTTP layer.

Endpoints:
  POST /ingest          — upload & index a document
  POST /query           — synchronous Q&A
  POST /query/stream    — streaming Q&A (Server-Sent Events)
  POST /feedback        — submit thumbs-up/down signal
  GET  /health          — liveness probe
  GET  /stats           — index statistics
  POST /evaluate        — run evaluation pipeline
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.routers import ingest, query, evaluation, feedback
from config.settings import get_settings
from core.rag_service import RAGService

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_rag_service: RAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_service
    logger.info("Booting RAG service…")
    _rag_service = RAGService()

    # Auto-load persisted index if it exists
    index_path = Path(settings.index_dir)
    if (index_path / "dense.index").exists():
        try:
            _rag_service.load_index()
            logger.info("Loaded persisted index: %d chunks", _rag_service.chunk_count)
        except Exception as exc:
            logger.warning("Could not load index: %s", exc)

    app.state.rag = _rag_service
    yield
    logger.info("Shutting down RAG service.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="Multimodal RAG API",
        description="Production-grade retrieval-augmented generation over multimodal documents.",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request latency logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        logger.info("%s %s → %d (%.0f ms)", request.method, request.url.path, response.status_code, ms)
        return response

    app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
    app.include_router(query.router, prefix="/query", tags=["Query"])
    app.include_router(evaluation.router, prefix="/evaluate", tags=["Evaluation"])
    app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])

    @app.get("/health")
    async def health(request: Request):
        rag: RAGService = request.app.state.rag
        return {
            "status": "ok",
            "index_ready": rag.is_ready,
            "chunks": rag.chunk_count,
        }

    @app.get("/stats")
    async def stats(request: Request):
        rag: RAGService = request.app.state.rag
        return {
            "chunks_indexed": rag.chunk_count,
            "index_ready": rag.is_ready,
            "cache_stats": rag._cache.stats,
            "settings": {
                "llm_model": settings.llm_model,
                "embedding_model": settings.embedding_model,
                "hybrid_alpha": settings.hybrid_alpha,
                "reranking_enabled": settings.enable_reranking,
                "query_rewriting_enabled": settings.enable_query_rewriting,
            },
        }

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers,
    )
