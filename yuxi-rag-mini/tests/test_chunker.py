"""Test chunking: short text, long text, overlap, metadata."""
import pytest
from app.rag.chunking.text_chunker import chunk_text, count_tokens, ChunkResult
from app.rag.chunking.markdown_chunker import chunk_markdown


def test_short_text_creates_one_chunk():
    result = chunk_text("Hello world", chunk_size=512, file_id="f1", kb_id="kb1", filename="test.txt")
    assert len(result) == 1
    assert result[0].content == "Hello world"
    assert result[0].chunk_index == 0
    assert result[0].file_id == "f1"
    assert result[0].kb_id == "kb1"


def test_long_text_creates_multiple_chunks():
    # Create text that exceeds chunk_size
    long_text = "\n".join([f"Line {i} with some content." for i in range(100)])
    result = chunk_text(long_text, chunk_size=20, file_id="f1", kb_id="kb1")
    assert len(result) > 1


def test_overlap_works():
    long_text = "\n".join([f"Word{i}" for i in range(50)])
    result_no_overlap = chunk_text(long_text, chunk_size=10, chunk_overlap=0, file_id="f1")
    result_with_overlap = chunk_text(long_text, chunk_size=10, chunk_overlap=3, file_id="f2")
    # With overlap, we should get more chunks or same number but with overlapping content
    assert len(result_with_overlap) >= len(result_no_overlap)


def test_metadata_preserved():
    meta = {"key1": "value1", "page_number": 3}
    result = chunk_text("Hello world", chunk_size=512, file_id="f1", kb_id="kb1",
                        filename="test.txt", metadata=meta)
    assert len(result) == 1
    assert result[0].metadata["key1"] == "value1"
    assert result[0].metadata["page_number"] == 3


def test_empty_text_returns_empty():
    result = chunk_text("", chunk_size=512)
    assert len(result) == 0


def test_whitespace_only_returns_empty():
    result = chunk_text("   \n  \n  ", chunk_size=512)
    assert len(result) == 0


def test_markdown_chunking_preserves_headings():
    md_text = "# Title 1\n\nParagraph 1\n\n# Title 2\n\nParagraph 2"
    result = chunk_markdown(md_text, chunk_size=512, file_id="f1", kb_id="kb1")
    assert len(result) >= 1
    # Should contain heading text
    all_content = " ".join(r.content for r in result)
    assert "Title 1" in all_content
    assert "Title 2" in all_content


def test_count_tokens():
    assert count_tokens("") == 0
    assert count_tokens("hello") >= 1
    assert count_tokens("hello world") >= 2
    # Chinese characters
    assert count_tokens("你好世界") >= 4


def test_chunk_result_dataclass():
    chunk = ChunkResult(
        content="test content",
        chunk_index=0,
        chunk_id="chunk_0",
        file_id="f1",
        kb_id="kb1",
        filename="test.txt",
        metadata={"page": 1},
    )
    assert chunk.content == "test content"
    assert chunk.metadata["page"] == 1
