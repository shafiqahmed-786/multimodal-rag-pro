"""
Ingestion pipeline — orchestrates parsing, chunking, and indexing.

Supports:
- Multiple file formats via pluggable parsers
- Async-friendly batch processing
- Progress callbacks for UI updates
- Duplicate detection via content hashing
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable, Optional

from core.interfaces import BaseChunker, BaseDocumentParser, BaseRetriever
from core.models import Chunk, RawPage
from ingestion.pdf_parser import PdfTextParser
from ingestion.table_parser import TableParser
from ingestion.image_processor import ImageOcrParser

logger = logging.getLogger(__name__)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class IngestionPipeline:
    """
    Orchestrates the full document ingestion flow:
    parse → chunk → index.

    Parsers are registered per file extension. Multiple parsers
    can run on the same file (e.g., text + table + OCR for PDFs).
    """

    def __init__(
        self,
        chunker: BaseChunker,
        retriever: BaseRetriever,
        extra_parsers: Optional[list[BaseDocumentParser]] = None,
    ) -> None:
        self.chunker = chunker
        self.retriever = retriever

        # Default parsers applied to every supported file
        self._parsers: list[BaseDocumentParser] = [
            PdfTextParser(),
            TableParser(),
            ImageOcrParser(),
        ]
        if extra_parsers:
            self._parsers.extend(extra_parsers)

        self._seen_hashes: set[str] = set()   # deduplication

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        file_path: str | Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> list[Chunk]:
        """
        Ingest a single file end-to-end.
        Returns the list of chunks that were indexed.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info("Ingesting: %s", path.name)
        _progress(progress_callback, f"Parsing {path.name}…", 0.1)

        # 1. Parse with all applicable parsers
        all_pages: list[RawPage] = []
        for parser in self._parsers:
            if parser.supports(str(path)):
                try:
                    pages = parser.parse(str(path))
                    all_pages.extend(pages)
                    logger.debug(
                        "%s → %d pages via %s",
                        path.name, len(pages), type(parser).__name__,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Parser %s failed: %s", type(parser).__name__, exc)

        if not all_pages:
            logger.warning("No content extracted from %s", path.name)
            return []

        _progress(progress_callback, "Chunking…", 0.4)

        # 2. Chunk
        chunks = self.chunker.chunk(all_pages)
        logger.info("Generated %d chunks from %s", len(chunks), path.name)

        # 3. Deduplicate
        unique_chunks = self._deduplicate(chunks)
        _progress(progress_callback, "Indexing…", 0.7)

        # 4. Index
        self.retriever.index(unique_chunks)
        _progress(progress_callback, "Done", 1.0)

        logger.info("Indexed %d unique chunks.", len(unique_chunks))
        return unique_chunks

    def ingest_directory(
        self,
        directory: str | Path,
        glob: str = "**/*.pdf",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> dict[str, list[Chunk]]:
        """Ingest all matching files in a directory."""
        directory = Path(directory)
        files = list(directory.glob(glob))
        results: dict[str, list[Chunk]] = {}

        for i, file_path in enumerate(files):
            base_progress = i / max(len(files), 1)

            def scoped_callback(msg: str, pct: float) -> None:
                if progress_callback:
                    overall = base_progress + pct / max(len(files), 1)
                    progress_callback(f"[{file_path.name}] {msg}", overall)

            chunks = self.ingest_file(file_path, scoped_callback)
            results[str(file_path)] = chunks

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _deduplicate(self, chunks: list[Chunk]) -> list[Chunk]:
        unique: list[Chunk] = []
        for chunk in chunks:
            h = _content_hash(chunk.content)
            if h not in self._seen_hashes:
                self._seen_hashes.add(h)
                unique.append(chunk)
        return unique

    def reset(self) -> None:
        """Clear deduplication state (useful between separate ingestion runs)."""
        self._seen_hashes.clear()


def _progress(
    callback: Optional[Callable[[str, float], None]],
    message: str,
    fraction: float,
) -> None:
    if callback:
        try:
            callback(message, fraction)
        except Exception:  # noqa: BLE001
            pass
