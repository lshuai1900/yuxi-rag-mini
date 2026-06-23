"""Test KnowledgeChunkModel has graph-related fields."""
import pytest
from app.rag.storage.models import KnowledgeChunkModel


def test_chunk_model_has_graph_fields():
    """Verify graph fields exist on the model."""
    columns = {c.name for c in KnowledgeChunkModel.__table__.columns}
    assert "start_token_pos" in columns
    assert "end_token_pos" in columns
    assert "graph_indexed" in columns
    assert "ent_ids" in columns
    assert "tags" in columns
    assert "extraction_result" in columns


def test_chunk_model_core_fields():
    """Verify core fields still exist."""
    columns = {c.name for c in KnowledgeChunkModel.__table__.columns}
    assert "chunk_id" in columns
    assert "file_id" in columns
    assert "kb_id" in columns
    assert "chunk_index" in columns
    assert "content" in columns
    assert "start_char_pos" in columns
    assert "end_char_pos" in columns
