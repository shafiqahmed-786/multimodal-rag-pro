"""
Layout-aware PDF text parser using PyMuPDF.

Improvements over the original:
- Detects headings from font-size heuristics
- Preserves reading order via block sorting
- Cleans artefacts (ligatures, hyphenation, control chars)
- Emits structured RawPage objects with section metadata
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

from core.interfaces import BaseDocumentParser
from core.models import ContentType, RawPage
from config.settings import get_settings

settings = get_settings()


# Regex patterns for common PDF artefacts
_LIGATURE_MAP = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st",
}
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def _clean_text(text: str) -> str:
    """Normalise Unicode, fix ligatures, remove control chars."""
    for lig, rep in _LIGATURE_MAP.items():
        text = text.replace(lig, rep)
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # re-join hyphenated words
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def _is_heading(span: dict, page_font_sizes: list[float]) -> bool:
    """Heuristic: a span is a heading if its font is ≥1.3× the median size."""
    if not page_font_sizes:
        return False
    sorted_sizes = sorted(page_font_sizes)
    median = sorted_sizes[len(sorted_sizes) // 2]
    is_large = span["size"] >= median * 1.3
    is_bold = "Bold" in span.get("font", "") or span.get("flags", 0) & 2**4
    return is_large or is_bold


class PdfTextParser(BaseDocumentParser):
    """
    Extracts text from PDFs using PyMuPDF, preserving structure.
    Returns one RawPage per PDF page, with section heading metadata.
    """

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(self, file_path: str) -> list[RawPage]:
        doc = fitz.open(file_path)
        pages: list[RawPage] = []
        source_file = Path(file_path).name

        for page_idx, page in enumerate(doc):
            raw_dict = page.get_text("dict", sort=True)  # sort=True → reading order
            blocks = raw_dict.get("blocks", [])

            # Collect all font sizes for heading detection
            all_sizes: list[float] = [
                span["size"]
                for blk in blocks
                if blk.get("type") == 0          # text block
                for line in blk.get("lines", [])
                for span in line.get("spans", [])
            ]

            page_parts: list[str] = []
            current_heading: str | None = None
            headings_seen: list[str] = []

            for blk in blocks:
                if blk.get("type") != 0:         # skip image blocks (handled by OCR)
                    continue

                block_text_parts = []
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        if _is_heading(span, all_sizes):
                            # Flush accumulated text before heading
                            if block_text_parts:
                                page_parts.append(" ".join(block_text_parts))
                                block_text_parts = []
                            current_heading = text
                            headings_seen.append(text)
                            page_parts.append(f"\n## {text}\n")
                        else:
                            block_text_parts.append(text)

                if block_text_parts:
                    page_parts.append(" ".join(block_text_parts))

            full_text = _clean_text("\n".join(page_parts))

            if not full_text:
                continue

            pages.append(
                RawPage(
                    page_number=page_idx + 1,
                    content=full_text,
                    content_type=ContentType.TEXT,
                    source_file=source_file,
                    metadata={
                        "headings": headings_seen,
                        "current_heading": current_heading,
                        "char_count": len(full_text),
                    },
                )
            )

        doc.close()
        return pages
