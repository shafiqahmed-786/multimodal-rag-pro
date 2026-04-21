"""
Table parser using pdfplumber — upgraded to dual representation.

Key improvement: tables are stored with BOTH:
  - semantic prose content (for dense + BM25 retrieval)
  - structured JSON (injected into LLM prompt for exact value lookups)

This eliminates the core failure mode where pipe-formatted markdown tables
score poorly on both dense and sparse retrieval axes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pdfplumber

from core.interfaces import BaseDocumentParser
from core.models import ContentType, RawPage

logger = logging.getLogger(__name__)


def _rows_to_markdown(rows: list[list[Optional[str]]]) -> str:
    """Convert a 2-D list of cells to a Markdown table string."""
    if not rows:
        return ""

    cleaned: list[list[str]] = []
    for row in rows:
        cleaned.append([str(cell).strip() if cell is not None else "" for cell in row])

    if not cleaned:
        return ""

    header = cleaned[0]
    body = cleaned[1:]
    body = [r for r in body if r != header]

    col_widths = [max(len(h), 3) for h in header]
    for row in body:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(row: list[str]) -> str:
        padded = [
            cell.ljust(col_widths[i]) if i < len(col_widths) else cell
            for i, cell in enumerate(row)
        ]
        return "| " + " | ".join(padded) + " |"

    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [fmt_row(header), separator] + [fmt_row(row) for row in body]
    return "\n".join(lines)


def _build_dual_representation(
    rows: list[list[Optional[str]]],
    page: int,
    source: str,
    table_index: int,
) -> tuple[str, str]:
    """
    Build two representations of a table:

    Returns (semantic_text, structured_json):
      - semantic_text: natural language prose for retrieval (BM25 + dense)
      - structured_json: JSON string for LLM exact-value lookup
    """
    if not rows or len(rows) < 2:
        return "", ""

    # Sanitise
    cleaned = [[str(c).strip() if c is not None else "" for c in row] for row in rows]
    headers = cleaned[0]
    body_rows = [r for r in cleaned[1:] if r != headers]

    # --- Structured JSON (for LLM prompt) ---
    records = []
    for row in body_rows:
        record: dict[str, str] = {}
        for i, cell in enumerate(row):
            key = headers[i] if i < len(headers) else f"col_{i}"
            record[key] = cell
        records.append(record)

    structured_json = json.dumps({"headers": headers, "rows": records}, indent=2)

    # --- Semantic prose (for retrieval) ---
    lines = [
        f"Table {table_index} from page {page} of {source}.",
        f"Columns: {', '.join(h for h in headers if h)}.",
    ]
    for rec in records:
        row_parts = [f"{k}: {v}" for k, v in rec.items() if v]
        if row_parts:
            lines.append(". ".join(row_parts) + ".")

    semantic_text = "\n".join(lines)
    return semantic_text, structured_json


class TableParser(BaseDocumentParser):
    """
    Extracts tables from PDFs using pdfplumber.
    Returns one RawPage per table found (not per PDF page).

    Each RawPage carries:
      - content       : semantic prose (for retrieval)
      - metadata.structured_data : JSON string (injected into LLM prompt)
      - metadata.markdown        : markdown fallback (for display)
    """

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(self, file_path: str) -> list[RawPage]:
        raw_pages: list[RawPage] = []
        source_file = Path(file_path).name
        table_index = 0

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    tables = page.extract_tables(
                        table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "snap_tolerance": 3,
                        }
                    )

                    if not tables:
                        tables = page.extract_tables()

                    if not tables:
                        continue

                    for table in tables:
                        if not table or len(table) < 2:
                            continue

                        table_index += 1
                        semantic_text, structured_json = _build_dual_representation(
                            table, page_idx + 1, source_file, table_index
                        )

                        if not semantic_text.strip():
                            continue

                        markdown = _rows_to_markdown(table)

                        raw_pages.append(
                            RawPage(
                                page_number=page_idx + 1,
                                content=semantic_text,        # ← retrieval-friendly prose
                                content_type=ContentType.TABLE,
                                source_file=source_file,
                                metadata={
                                    "table_index": table_index,
                                    "rows": len(table),
                                    "cols": max(len(r) for r in table),
                                    "structured_data": structured_json,  # ← LLM lookup
                                    "markdown": markdown,                 # ← display fallback
                                },
                            )
                        )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Table extraction failed for %s: %s", file_path, exc)

        return raw_pages
