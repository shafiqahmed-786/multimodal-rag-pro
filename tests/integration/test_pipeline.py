"""
Integration tests — full ingestion pipeline on the sample PDF.
These tests require the sample PDF to be present.
Skip if not found (CI without assets).
"""
import os
import pytest
from pathlib import Path

SAMPLE_PDF = Path(__file__).parents[2] / "data" / "raw" / "qatar_test_doc.pdf"
HAS_PDF = SAMPLE_PDF.exists()


@pytest.mark.skipif(not HAS_PDF, reason="Sample PDF not found")
class TestIngestionIntegration:
    @pytest.fixture(scope="class")
    def ingested_chunks(self):
        from chunking.semantic_chunker import SemanticChunker
        from retrieval.hybrid_retriever import HybridRetriever
        from ingestion.pipeline import IngestionPipeline

        chunker = SemanticChunker(chunk_size=300, chunk_overlap=50)
        retriever = HybridRetriever()
        pipeline = IngestionPipeline(chunker=chunker, retriever=retriever)
        return pipeline.ingest_file(str(SAMPLE_PDF))

    def test_returns_chunks(self, ingested_chunks):
        assert len(ingested_chunks) > 0

    def test_chunks_have_content(self, ingested_chunks):
        for c in ingested_chunks:
            assert c.content.strip()

    def test_chunks_have_page_numbers(self, ingested_chunks):
        assert all(c.page_number >= 1 for c in ingested_chunks)

    def test_chunks_have_source_file(self, ingested_chunks):
        assert all(c.source_file for c in ingested_chunks)

    def test_multiple_content_types(self, ingested_chunks):
        types = {c.content_type for c in ingested_chunks}
        # At minimum should have text
        from core.models import ContentType
        assert ContentType.TEXT in types

    def test_no_empty_chunks(self, ingested_chunks):
        from config.settings import get_settings
        min_len = get_settings().min_chunk_length
        assert all(len(c.content) >= min_len for c in ingested_chunks)


@pytest.mark.skipif(not HAS_PDF, reason="Sample PDF not found")
class TestRetrievalIntegration:
    @pytest.fixture(scope="class")
    def loaded_retriever(self):
        from chunking.semantic_chunker import SemanticChunker
        from retrieval.hybrid_retriever import HybridRetriever
        from ingestion.pipeline import IngestionPipeline

        chunker = SemanticChunker(chunk_size=300, chunk_overlap=50)
        retriever = HybridRetriever()
        pipeline = IngestionPipeline(chunker=chunker, retriever=retriever)
        pipeline.ingest_file(str(SAMPLE_PDF))
        return retriever

    def test_retrieves_results(self, loaded_retriever):
        results = loaded_retriever.retrieve("What is Qatar?", top_k=3)
        assert len(results) >= 1

    def test_results_have_scores(self, loaded_retriever):
        results = loaded_retriever.retrieve("economy", top_k=3)
        assert all(r.score >= 0 for r in results)

    def test_results_sorted_by_score(self, loaded_retriever):
        results = loaded_retriever.retrieve("education", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
