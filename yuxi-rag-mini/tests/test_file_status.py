import pytest
from app.rag.base import FileStatus, KnowledgeBase


def test_file_status_values():
    assert FileStatus.UPLOADED == "uploaded"
    assert FileStatus.PARSING == "parsing"
    assert FileStatus.PARSED == "parsed"
    assert FileStatus.ERROR_PARSING == "error_parsing"
    assert FileStatus.INDEXING == "indexing"
    assert FileStatus.INDEXED == "indexed"
    assert FileStatus.ERROR_INDEXING == "error_indexing"


def test_no_legacy_status():
    """Ensure legacy statuses are removed."""
    assert not hasattr(FileStatus, "CHUNKING")
    assert not hasattr(FileStatus, "EMBEDDING")
    assert not hasattr(FileStatus, "FAILED")


def test_is_terminal_status():
    assert KnowledgeBase.is_terminal_status(FileStatus.INDEXED) is True
    assert KnowledgeBase.is_terminal_status(FileStatus.ERROR_PARSING) is True
    assert KnowledgeBase.is_terminal_status(FileStatus.ERROR_INDEXING) is True
    assert KnowledgeBase.is_terminal_status(FileStatus.UPLOADED) is False
    assert KnowledgeBase.is_terminal_status(FileStatus.PARSING) is False
    assert KnowledgeBase.is_terminal_status(FileStatus.PARSED) is False
    assert KnowledgeBase.is_terminal_status(FileStatus.INDEXING) is False
