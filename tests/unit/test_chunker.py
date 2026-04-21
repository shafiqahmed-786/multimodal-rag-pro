"""Unit tests for SemanticChunker."""
import pytest
from core.models import ContentType, RawPage
from chunking.semantic_chunker import SemanticChunker, _split_by_headings


# ---------------------------------------------------------------------------
# Heading splitter
# ---------------------------------------------------------------------------

def test_split_by_headings_no_headings():
    text = "Just some plain text without any headings."
    result = _split_by_headings(text)
    assert len(result) == 1
    heading, body = result[0]
    assert heading is None
    assert body == text


def test_split_by_headings_with_headings():
    text = "Preamble text.\n## Section One\nContent one.\n## Section Two\nContent two."
    result = _split_by_headings(text)
    assert len(result) == 3
    assert result[0][0] is None
    assert result[1][0] == "Section One"
    assert result[2][0] == "Section Two"


def test_split_by_headings_only_heading():
    text = "## Only Heading\nSome content."
    result = _split_by_headings(text)
    assert result[0][0] == "Only Heading"


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

@pytest.fixture
def chunker():
    return SemanticChunker(chunk_size=200, chunk_overlap=20, min_chunk_length=10)


def make_page(content, ctype=ContentType.TEXT, page=1):
    return RawPage(
        page_number=page,
        content=content,
        content_type=ctype,
        source_file="test.pdf",
    )


def test_chunker_returns_list(chunker):
    pages = [make_page("Hello world. This is a short text.")]
    chunks = chunker.chunk(pages)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chunker_preserves_source(chunker):
    pages = [make_page("Some content here.", page=3)]
    chunks = chunker.chunk(pages)
    assert all(c.source_file == "test.pdf" for c in chunks)
    assert all(c.page_number == 3 for c in chunks)


def test_table_not_split(chunker):
    """Tables must never be split regardless of size."""
    long_table = "TABLE 1:\n" + "| col1 | col2 |\n|------|------|\n" + "| a    | b    |\n" * 50
    pages = [make_page(long_table, ctype=ContentType.TABLE)]
    chunks = chunker.chunk(pages)
    assert len(chunks) == 1
    assert chunks[0].content_type == ContentType.TABLE


def test_image_chunk(chunker):
    pages = [make_page("OCR extracted text from image", ctype=ContentType.IMAGE)]
    chunks = chunker.chunk(pages)
    assert len(chunks) == 1
    assert chunks[0].content_type == ContentType.IMAGE


def test_min_length_filter(chunker):
    pages = [make_page("Hi")]   # shorter than min_chunk_length=10
    chunks = chunker.chunk(pages)
    assert len(chunks) == 0


def test_heading_propagated(chunker):
    text = "## Introduction\nThis is the introduction section with enough content to form a chunk."
    pages = [make_page(text)]
    chunks = chunker.chunk(pages)
    assert any(c.section_heading == "Introduction" for c in chunks)


def test_multiple_pages(chunker):
    pages = [
        make_page("Page one content with some text.", page=1),
        make_page("Page two content with different text.", page=2),
    ]
    chunks = chunker.chunk(pages)
    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers


def test_empty_pages(chunker):
    chunks = chunker.chunk([])
    assert chunks == []
