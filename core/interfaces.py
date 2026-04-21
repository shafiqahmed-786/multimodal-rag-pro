"""
Abstract interfaces (protocols) for every swappable component.
This enforces clean boundaries and makes each layer independently testable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from core.models import (
    Answer,
    Chunk,
    IndexedChunk,
    QueryExpansion,
    QueryRequest,
    RawPage,
    RetrievedChunk,
    VerificationResult,
)


class BaseDocumentParser(ABC):
    """Extracts raw page content from a document."""

    @abstractmethod
    def parse(self, file_path: str) -> list[RawPage]:
        """Parse a document and return a list of raw pages."""
        ...

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Return True if this parser can handle the given file."""
        ...


class BaseChunker(ABC):
    """Splits raw pages into atomic, retrievable chunks."""

    @abstractmethod
    def chunk(self, pages: list[RawPage]) -> list[Chunk]:
        """Split pages into chunks."""
        ...


class BaseEncoder(ABC):
    """Produces vector representations of text."""

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into embedding vectors."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class BaseRetriever(ABC):
    """Retrieves relevant chunks for a query."""

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Build or update the retrieval index."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve top-k relevant chunks for a query."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a persisted index from disk."""
        ...


class BaseQueryProcessor(ABC):
    """Rewrites and expands queries for better retrieval."""

    @abstractmethod
    def process(self, query: str) -> QueryExpansion:
        """Rewrite and expand a query."""
        ...


class BaseReranker(ABC):
    """Re-scores retrieved chunks using a cross-encoder."""

    @abstractmethod
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks."""
        ...


class BaseGenerator(ABC):
    """Generates answers from retrieved context."""

    @abstractmethod
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Generate an answer string."""
        ...

    @abstractmethod
    async def stream(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> AsyncIterator[str]:
        """Stream an answer token by token."""
        ...


class BaseVerifier(ABC):
    """Verifies that an answer is faithful to its sources."""

    @abstractmethod
    def verify(
        self, query: str, answer: str, chunks: list[RetrievedChunk]
    ) -> VerificationResult:
        """Check for hallucination / faithfulness."""
        ...


class BaseCacheManager(ABC):
    """Key-value cache abstraction."""

    @abstractmethod
    def get(self, key: str):
        ...

    @abstractmethod
    def set(self, key: str, value, ttl: int | None = None) -> None:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...
