"""
API schemas — Pydantic models for request/response validation.
Separate from core domain models to allow API versioning.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.models import AnswerConfidence, ContentType, RetrievalMode


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    file: str
    chunks_indexed: int
    message: str = "Document ingested successfully."


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class QueryPayload(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    enable_reranking: bool = True
    enable_query_rewriting: bool = True
    stream: bool = False


class CitationOut(BaseModel):
    source_file: str
    page_number: int
    content_type: ContentType
    excerpt: str
    relevance_score: float


class RetrievedChunkOut(BaseModel):
    content: str
    source_file: str
    page_number: int
    content_type: ContentType
    section_heading: Optional[str]
    score: float
    rank: int
    retrieval_method: str


class VerificationOut(BaseModel):
    is_faithful: bool
    confidence: AnswerConfidence
    issues: list[str]
    verification_note: str


class QueryExpansionOut(BaseModel):
    original: str
    rewritten: str
    sub_queries: list[str]


class AnswerResponse(BaseModel):
    id: str
    query_id: str
    text: str
    confidence: AnswerConfidence
    citations: list[CitationOut]
    retrieved_chunks: list[RetrievedChunkOut]
    query_expansion: Optional[QueryExpansionOut]
    verification: Optional[VerificationOut]
    latency_ms: float
    retrieval_debug: dict[str, Any]
    created_at: datetime


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackPayload(BaseModel):
    answer_id: str
    query_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    accepted: bool
    message: str


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvalSampleIn(BaseModel):
    question: str
    ground_truth: str
    predicted_answer: str
    retrieved_contexts: list[str]


class EvalRequest(BaseModel):
    samples: list[EvalSampleIn]


class EvalResponse(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    overall_score: float
    samples_evaluated: int
    evaluated_at: datetime
