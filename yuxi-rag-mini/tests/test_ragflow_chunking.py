import pytest
from app.rag.chunking.ragflow_like.nlp import count_tokens, naive_merge, hard_split_by_token_limit
from app.rag.chunking.ragflow_like.presets import normalize_chunk_preset_id, resolve_chunk_processing_params
from app.rag.chunking.ragflow_like.dispatcher import chunk_markdown


def test_count_tokens_english():
    assert count_tokens("hello world") == 2


def test_count_tokens_chinese():
    assert count_tokens("你好世界") == 4


def test_count_tokens_mixed():
    # "hello" is one word token, "你" and "好" are two char tokens
    assert count_tokens("hello 你好") == 3


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_naive_merge_basic():
    sections = ["line1", "line2", "line3"]
    result = naive_merge(sections, chunk_token_num=2)
    assert len(result) >= 1
    # Each chunk should have approximately chunk_token_num tokens


def test_naive_merge_single_section():
    result = naive_merge(["short text"], chunk_token_num=100)
    assert len(result) == 1
    assert result[0] == "short text"


def test_hard_split_by_token_limit():
    text = "word " * 100
    result = hard_split_by_token_limit(text, chunk_token_num=20, hard_limit_token_num=20)
    assert len(result) > 1


def test_hard_split_no_limit():
    text = "some text"
    result = hard_split_by_token_limit(text, chunk_token_num=10, hard_limit_token_num=0)
    assert result == [text]


def test_normalize_chunk_preset_id():
    assert normalize_chunk_preset_id("general") == "general"
    assert normalize_chunk_preset_id("unknown") == "general"
    assert normalize_chunk_preset_id("") == "general"
    assert normalize_chunk_preset_id("book") == "book"


def test_resolve_chunk_processing_params_default():
    params = resolve_chunk_processing_params()
    assert params["chunk_preset_id"] == "general"
    assert params["chunk_token_num"] == 512


def test_resolve_chunk_processing_params_book():
    params = resolve_chunk_processing_params(preset_id="book")
    assert params["chunk_token_num"] == 1024


def test_dispatcher_chunk_markdown():
    markdown = "# Title\n\nParagraph 1 with some content.\n\n## Section\n\nMore content here."
    chunks = chunk_markdown(markdown, file_id="test_file", filename="test.md")
    assert len(chunks) >= 1
    assert all("content" in c for c in chunks)
    assert all("chunk_id" in c for c in chunks)
    assert all("file_id" in c for c in chunks)
    assert all(c["file_id"] == "test_file" for c in chunks)


def test_dispatcher_chunk_markdown_empty():
    chunks = chunk_markdown("", file_id="test", filename="empty.md")
    assert chunks == []


def test_dispatcher_chunk_markdown_with_params():
    markdown = "Some content " * 100
    chunks = chunk_markdown(markdown, file_id="test", filename="test.txt",
                           processing_params={"chunk_preset_id": "general"})
    assert len(chunks) >= 1
