"""Unit tests for ingestion parsers."""
import pytest
from ingestion.table_parser import _rows_to_markdown
from ingestion.pdf_parser import _clean_text, _is_heading


# ---------------------------------------------------------------------------
# Table markdown conversion
# ---------------------------------------------------------------------------

def test_rows_to_markdown_basic():
    rows = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
    md = _rows_to_markdown(rows)
    assert "Name" in md
    assert "Alice" in md
    assert "|" in md
    assert "---" in md


def test_rows_to_markdown_empty():
    assert _rows_to_markdown([]) == ""


def test_rows_to_markdown_single_row():
    rows = [["Header"]]
    md = _rows_to_markdown(rows)
    # Single row → header only, body empty
    assert "Header" in md


def test_rows_to_markdown_none_cells():
    rows = [["Col1", None], [None, "Value"]]
    md = _rows_to_markdown(rows)
    assert "Col1" in md
    # None cells should become empty strings, not crash
    assert "None" not in md


def test_rows_to_markdown_deduplicates_header():
    rows = [["A", "B"], ["A", "B"], ["1", "2"]]
    md = _rows_to_markdown(rows)
    lines = md.split("\n")
    # Header row should not appear twice in body
    header_rows = [l for l in lines if "A" in l and "B" in l and "---" not in l]
    assert len(header_rows) == 1


# ---------------------------------------------------------------------------
# PDF text cleaning
# ---------------------------------------------------------------------------

def test_clean_text_ligatures():
    # ﬁ ligature should become "fi"
    assert _clean_text("\ufb01le") == "file"


def test_clean_text_hyphen_break():
    result = _clean_text("algo-\nrithm")
    assert "algorithm" in result


def test_clean_text_multiple_newlines():
    result = _clean_text("a\n\n\n\n\nb")
    assert "\n\n\n" not in result


def test_clean_text_empty():
    assert _clean_text("") == ""


def test_clean_text_control_chars():
    result = _clean_text("hello\x00world")
    assert "\x00" not in result


# ---------------------------------------------------------------------------
# Heading detection heuristic
# ---------------------------------------------------------------------------

def test_is_heading_large_font():
    span = {"size": 24.0, "font": "Arial", "flags": 0, "text": "Chapter 1"}
    page_sizes = [10.0, 10.0, 11.0, 24.0]
    assert _is_heading(span, page_sizes) is True


def test_is_heading_normal_font():
    span = {"size": 11.0, "font": "Arial", "flags": 0, "text": "regular text"}
    page_sizes = [10.0, 11.0, 11.0, 11.0]
    assert _is_heading(span, page_sizes) is False


def test_is_heading_bold():
    span = {"size": 11.0, "font": "Arial-Bold", "flags": 0, "text": "bold text"}
    page_sizes = [10.0, 11.0, 11.0]
    assert _is_heading(span, page_sizes) is True


def test_is_heading_empty_sizes():
    span = {"size": 14.0, "font": "Arial", "flags": 0}
    assert _is_heading(span, []) is False
