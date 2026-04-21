"""
Semantic chunker that respects document structure.

Strategy:
1. Split on headings first (preserves document sections)
2. Then apply RecursiveCharacterTextSplitter within each section
3. Carry heading metadata into every child chunk
4. Filter out chunks that are too short to be meaningful
"""
from __future__ import annotations

import re
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.interfaces import BaseChunker
from core.models import Chunk, ContentType, RawPage
from config.settings import get_settings

settings = get_settings()

_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def _split_by_headings(text: str) -> list[tuple[Optional[str], str]]:
    """
    Split text into (heading, body) pairs.
    Returns [(None, preamble), (heading1, body1), ...]
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[Optional[str], str]] = []
    # Preamble before first heading
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append((None, preamble))

    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append((heading, body))

    return sections


class SemanticChunker(BaseChunker):
    """
    Chunks documents semantically by:
    - Respecting section/heading boundaries
    - Keeping tables intact (no mid-table splits)
    - Applying character-level splitting within sections
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_length: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_length = min_chunk_length or settings.min_chunk_length

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, pages: list[RawPage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_idx = 0

        for page in pages:
            if page.content_type == ContentType.TABLE:
                # Tables stay whole — never split mid-table
                chunk = Chunk(
                    content=page.content,
                    content_type=page.content_type,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=chunk_idx,
                    section_heading=page.metadata.get("current_heading"),
                    metadata=page.metadata,
                )
                if len(chunk.content) >= self.min_chunk_length:
                    chunks.append(chunk)
                    chunk_idx += 1

            elif page.content_type == ContentType.IMAGE:
                # OCR text — treat as a single chunk
                chunk = Chunk(
                    content=page.content,
                    content_type=page.content_type,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=chunk_idx,
                    metadata=page.metadata,
                )
                if len(chunk.content) >= self.min_chunk_length:
                    chunks.append(chunk)
                    chunk_idx += 1

            else:
                # Regular text — split by headings then by size
                sections = _split_by_headings(page.content)
                current_heading = page.metadata.get("current_heading")

                for heading, body in sections:
                    active_heading = heading or current_heading
                    if heading:
                        current_heading = heading

                    splits = self._splitter.split_text(body)
                    for split in splits:
                        if len(split.strip()) < self.min_chunk_length:
                            continue
                        chunks.append(
                            Chunk(
                                content=split.strip(),
                                content_type=ContentType.TEXT,
                                source_file=page.source_file,
                                page_number=page.page_number,
                                chunk_index=chunk_idx,
                                section_heading=active_heading,
                                metadata={
                                    **page.metadata,
                                    "section": active_heading,
                                },
                            )
                        )
                        chunk_idx += 1

        return chunks
