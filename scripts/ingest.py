#!/usr/bin/env python3
"""
CLI script — ingest one or more PDFs and persist the index.

Usage:
    python scripts/ingest.py data/raw/my_doc.pdf
    python scripts/ingest.py data/raw/*.pdf --index-dir data/indexes
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from config.settings import get_settings
from chunking.semantic_chunker import SemanticChunker
from retrieval.hybrid_retriever import HybridRetriever
from ingestion.pipeline import IngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ingest")
settings = get_settings()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG index.")
    parser.add_argument("files", nargs="+", help="PDF file paths to ingest")
    parser.add_argument(
        "--index-dir",
        default=settings.index_dir,
        help=f"Directory to save the index (default: {settings.index_dir})",
    )
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    args = parser.parse_args()

    chunker = SemanticChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    retriever = HybridRetriever()

    # Load existing index if present
    index_path = Path(args.index_dir)
    if (index_path / "dense.index").exists():
        logger.info("Loading existing index from %s", index_path)
        retriever.load(str(index_path))

    pipeline = IngestionPipeline(chunker=chunker, retriever=retriever)

    total_chunks = 0
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            logger.error("File not found: %s", path)
            continue

        def progress(msg: str, frac: float) -> None:
            bar = "█" * int(frac * 20) + "░" * (20 - int(frac * 20))
            print(f"\r  [{bar}] {frac*100:.0f}% {msg}", end="", flush=True)

        chunks = pipeline.ingest_file(str(path), progress_callback=progress)
        print()   # newline after progress bar
        logger.info("✓ %s → %d chunks", path.name, len(chunks))
        total_chunks += len(chunks)

    retriever.save(args.index_dir)
    logger.info(
        "Index saved to %s | total chunks: %d | total in index: %d",
        args.index_dir, total_chunks, retriever.chunk_count,
    )


if __name__ == "__main__":
    main()
