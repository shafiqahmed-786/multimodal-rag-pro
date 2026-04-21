"""
Qdrant vector store — drop-in replacement for the FAISS dense retriever.

Why Qdrant over FAISS for production:
  - Persistent by default (no pickle hacks)
  - Metadata filtering (by page, file, content_type)
  - Horizontal scaling & cloud hosting (Qdrant Cloud)
  - Payload search + ANN in one query
  - Native payload filtering reduces re-ranking overhead

Usage:
    pip install qdrant-client
    Set VECTOR_STORE=qdrant in .env.
    Then swap HybridRetriever._dense for QdrantDenseRetriever.

Requires: qdrant-client>=1.9.0
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from core.models import Chunk, ContentType, RetrievedChunk
from embeddings.dense_encoder import DenseEncoder
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class QdrantDenseRetriever:
    """
    Dense retriever backed by Qdrant with metadata filtering support.
    """

    COLLECTION = "rag_chunks"

    def __init__(
        self,
        encoder: Optional[DenseEncoder] = None,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
    ) -> None:
        self._encoder = encoder or DenseEncoder()
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = QdrantClient(url=url, api_key=api_key)
            self._client.recreate_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(
                    size=self._encoder.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Qdrant collection '%s' ready at %s", self.COLLECTION, url)
        except Exception as exc:
            logger.warning("Qdrant unavailable: %s — falling back to FAISS.", exc)
            self._client = None

    def index(self, chunks: list[Chunk]) -> None:
        if self._client is None:
            return
        from qdrant_client.models import PointStruct

        texts = [c.content for c in chunks]
        vectors = self._encoder.encode(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "content_type": chunk.content_type.value,
                    "section_heading": chunk.section_heading,
                },
            )
            for chunk, vec in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self.COLLECTION, points=points)
        logger.info("Upserted %d points to Qdrant.", len(points))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        if self._client is None:
            return []

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        q_vec = self._encoder.encode([query])[0]

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self.COLLECTION,
            query_vector=q_vec,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        retrieved = []
        for rank, hit in enumerate(results):
            p = hit.payload
            chunk = Chunk(
                id=p["chunk_id"],
                content=p["content"],
                content_type=ContentType(p["content_type"]),
                source_file=p["source_file"],
                page_number=p["page_number"],
                section_heading=p.get("section_heading"),
            )
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(hit.score),
                    retrieval_method="qdrant_dense",
                    rank=rank,
                )
            )
        return retrieved
