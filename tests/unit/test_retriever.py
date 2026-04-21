"""Unit tests for HybridRetriever."""
import os
import tempfile
import pytest

from core.models import Chunk, ContentType
from retrieval.hybrid_retriever import HybridRetriever, _rrf_fuse


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def test_rrf_fuse_merges_lists():
    dense = [(0, 0.9), (1, 0.7), (2, 0.5)]
    sparse = [(2, 0.8), (0, 0.6), (3, 0.4)]
    result = _rrf_fuse(dense, sparse)
    ids = [r[0] for r in result]
    # 0 appears in both → should rank highly
    assert ids[0] == 0


def test_rrf_fuse_empty_lists():
    result = _rrf_fuse([], [])
    assert result == []


def test_rrf_fuse_single_list():
    dense = [(0, 0.9), (1, 0.7)]
    result = _rrf_fuse(dense, [], alpha=1.0)
    assert result[0][0] == 0


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

def make_chunk(content: str, page: int = 1) -> Chunk:
    return Chunk(
        content=content,
        content_type=ContentType.TEXT,
        source_file="test.pdf",
        page_number=page,
    )


@pytest.fixture
def retriever():
    return HybridRetriever()


def test_retriever_index_and_retrieve(retriever):
    chunks = [
        make_chunk("The capital of France is Paris.", page=1),
        make_chunk("Python is a popular programming language.", page=2),
        make_chunk("Machine learning uses neural networks.", page=3),
    ]
    retriever.index(chunks)
    results = retriever.retrieve("What is the capital of France?", top_k=2)
    assert len(results) >= 1
    # First result should be about Paris
    top_content = results[0].chunk.content
    assert "Paris" in top_content or "France" in top_content


def test_retriever_top_k_respected(retriever):
    chunks = [make_chunk(f"Document about topic {i}") for i in range(10)]
    retriever.index(chunks)
    results = retriever.retrieve("topic", top_k=3)
    assert len(results) <= 3


def test_retriever_returns_scored_chunks(retriever):
    retriever.index([make_chunk("Some content")])
    results = retriever.retrieve("content")
    assert all(hasattr(r, "score") for r in results)
    assert all(hasattr(r, "rank") for r in results)


def test_retriever_persistence(retriever, tmp_path):
    chunks = [make_chunk("Persistent content"), make_chunk("Another chunk")]
    retriever.index(chunks)
    retriever.save(str(tmp_path))

    new_retriever = HybridRetriever()
    new_retriever.load(str(tmp_path))
    results = new_retriever.retrieve("persistent", top_k=1)
    assert len(results) == 1


def test_retriever_empty_index(retriever):
    results = retriever.retrieve("anything", top_k=5)
    assert results == []


def test_retriever_chunk_count(retriever):
    assert retriever.chunk_count == 0
    retriever.index([make_chunk("a"), make_chunk("b")])
    assert retriever.chunk_count == 2
