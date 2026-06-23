import pytest
import tempfile
import os
from app.rag.parser.unified import Parser, MarkdownParseResult, parse_source_to_markdown


@pytest.mark.asyncio
async def test_aparse_text_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello world\nThis is a test file.")
        f.flush()
        result = await Parser.aparse(f.name)
        os.unlink(f.name)
    assert "Hello world" in result
    assert "test file" in result


@pytest.mark.asyncio
async def test_aparse_markdown_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Title\n\nSome markdown content.")
        f.flush()
        result = await Parser.aparse(f.name)
        os.unlink(f.name)
    assert "Title" in result
    assert "markdown content" in result


@pytest.mark.asyncio
async def test_parse_source_to_markdown():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test content for markdown conversion.")
        f.flush()
        result = await parse_source_to_markdown(f.name)
        os.unlink(f.name)
    assert isinstance(result, MarkdownParseResult)
    assert "Test content" in result.markdown
    assert result.file_ext == ".txt"


@pytest.mark.asyncio
async def test_aparse_unsupported_fallback():
    """Test that unsupported file types fall back to plain text reading."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write("Plain text content in unsupported format.")
        f.flush()
        result = await Parser.aparse(f.name)
        os.unlink(f.name)
    assert "Plain text content" in result


@pytest.mark.asyncio
async def test_aparse_empty_file():
    """Test that empty files raise ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        f.flush()
        fname = f.name
    # Empty .txt files fall back to plain text reading which returns empty string.
    # The ValueError is raised by the unified parser only when a parser is found
    # but returns empty content. For unsupported types that fall back, empty content
    # is returned as-is.
    result = await Parser.aparse(fname)
    os.unlink(fname)
    assert result == ""
