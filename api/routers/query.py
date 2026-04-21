"""
/query router — synchronous and streaming Q&A endpoints.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.schemas.schemas import (
    AnswerResponse,
    CitationOut,
    QueryExpansionOut,
    QueryPayload,
    RetrievedChunkOut,
    VerificationOut,
)
from core.models import Answer, QueryRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _answer_to_response(answer: Answer) -> AnswerResponse:
    return AnswerResponse(
        id=answer.id,
        query_id=answer.query_id,
        text=answer.text,
        confidence=answer.confidence,
        citations=[
            CitationOut(
                source_file=c.source_file,
                page_number=c.page_number,
                content_type=c.content_type,
                excerpt=c.excerpt,
                relevance_score=c.relevance_score,
            )
            for c in answer.citations
        ],
        retrieved_chunks=[
            RetrievedChunkOut(
                content=rc.chunk.content,
                source_file=rc.chunk.source_file,
                page_number=rc.chunk.page_number,
                content_type=rc.chunk.content_type,
                section_heading=rc.chunk.section_heading,
                score=rc.score,
                rank=rc.rank,
                retrieval_method=rc.retrieval_method,
            )
            for rc in answer.retrieved_chunks
        ],
        query_expansion=QueryExpansionOut(
            original=answer.query_expansion.original,
            rewritten=answer.query_expansion.rewritten,
            sub_queries=answer.query_expansion.sub_queries,
        )
        if answer.query_expansion
        else None,
        verification=VerificationOut(
            is_faithful=answer.verification.is_faithful,
            confidence=answer.verification.confidence,
            issues=answer.verification.issues,
            verification_note=answer.verification.verification_note,
        )
        if answer.verification
        else None,
        latency_ms=answer.latency_ms,
        retrieval_debug=answer.retrieval_debug,
        created_at=answer.created_at,
    )


@router.post("", response_model=AnswerResponse)
async def query_sync(request: Request, payload: QueryPayload):
    """Synchronous Q&A — returns full answer with citations."""
    rag = request.app.state.rag
    if not rag.is_ready:
        raise HTTPException(status_code=503, detail="Index not ready. Ingest a document first.")

    qr = QueryRequest(
        text=payload.text,
        top_k=payload.top_k,
        retrieval_mode=payload.retrieval_mode,
        enable_reranking=payload.enable_reranking,
        enable_query_rewriting=payload.enable_query_rewriting,
    )
    try:
        answer = rag.query(qr)
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _answer_to_response(answer)


@router.post("/stream")
async def query_stream(request: Request, payload: QueryPayload):
    """
    Streaming Q&A via Server-Sent Events.
    Each event is a JSON object: {"token": "..."} or {"done": true}.
    """
    rag = request.app.state.rag
    if not rag.is_ready:
        raise HTTPException(status_code=503, detail="Index not ready.")

    qr = QueryRequest(
        text=payload.text,
        top_k=payload.top_k,
        enable_reranking=payload.enable_reranking,
        enable_query_rewriting=payload.enable_query_rewriting,
        stream=True,
    )

    async def event_generator():
        try:
            async for token in rag.query_stream(qr):
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            logger.exception("Stream failed: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
