"""
Core domain models for the Multimodal RAG system.
All data flowing through the pipeline is typed here.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    HEADING = "heading"
    CAPTION = "caption"


class RetrievalMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class AnswerConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"


# ---------------------------------------------------------------------------
# Document / Chunk models
# ---------------------------------------------------------------------------

class RawPage(BaseModel):
    """A single page extracted from a source document."""
    page_number: int
    content: str
    content_type: ContentType
    source_file: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """An atomic retrievable unit of content."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    content_type: ContentType
    source_file: str
    page_number: int
    chunk_index: int = 0
    section_heading: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Set during indexing
    embedding: Optional[list[float]] = None

    @property
    def citation_label(self) -> str:
        return f"[{self.source_file} | p.{self.page_number} | {self.content_type.value}]"


class IndexedChunk(Chunk):
    """Chunk with its FAISS index position."""
    faiss_id: int = 0


# ---------------------------------------------------------------------------
# Query / Retrieval models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Incoming query from the user."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    top_k: int = 5
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    filters: dict[str, Any] = Field(default_factory=dict)
    enable_reranking: bool = True
    enable_query_rewriting: bool = True
    stream: bool = False


class RetrievedChunk(BaseModel):
    """A chunk with its retrieval score."""
    chunk: Chunk
    score: float
    retrieval_method: str  # "dense", "sparse", "hybrid", "reranked:hybrid"
    rank: int
    retrieval_explanation: dict[str, Any] = Field(default_factory=dict)


class QueryExpansion(BaseModel):
    """Result of query rewriting."""
    original: str
    rewritten: str
    sub_queries: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Generation / Answer models
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A grounded citation mapping answer text to source chunks."""
    chunk_id: str
    source_file: str
    page_number: int
    content_type: ContentType
    excerpt: str  # Short excerpt from the chunk
    relevance_score: float


class VerificationResult(BaseModel):
    """Result of the hallucination self-check step."""
    is_faithful: bool
    confidence: AnswerConfidence
    issues: list[str] = Field(default_factory=list)
    verification_note: str = ""


class Answer(BaseModel):
    """The final answer returned to the user."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    query_expansion: Optional[QueryExpansion] = None
    verification: Optional[VerificationResult] = None
    confidence: AnswerConfidence = AnswerConfidence.UNVERIFIED
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Debug / explainability fields
    retrieval_debug: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------

class EvaluationSample(BaseModel):
    """A single QA pair for evaluation."""
    question: str
    ground_truth: str
    predicted_answer: str
    retrieved_contexts: list[str]


class EvaluationResult(BaseModel):
    """Aggregated evaluation metrics."""
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    overall_score: float
    samples_evaluated: int
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Feedback model
# ---------------------------------------------------------------------------

class FeedbackSignal(BaseModel):
    """User feedback on an answer (for future fine-tuning / RLHF)."""
    answer_id: str
    query_id: str
    rating: int  # 1 (thumbs down) or 5 (thumbs up)
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
