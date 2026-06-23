"""Test that all Phase 4 modules can be imported."""


def test_import_schemas():
    from app.rag.schemas import MilvusRetrievalConfig, QueryRequest, SearchResultSchema
    assert MilvusRetrievalConfig is not None


def test_import_base():
    from app.rag.base import FileStatus, KnowledgeBase
    assert FileStatus.ERROR_PARSING == "error_parsing"
    assert FileStatus.ERROR_INDEXING == "error_indexing"


def test_import_ragflow_chunking():
    from app.rag.chunking.ragflow_like import chunk_markdown
    from app.rag.chunking.ragflow_like.nlp import count_tokens, naive_merge
    from app.rag.chunking.ragflow_like.presets import resolve_chunk_processing_params
    from app.rag.chunking.ragflow_like.dispatcher import chunk_markdown as dispatcher_chunk
    assert chunk_markdown is not None


def test_import_unified_parser():
    from app.rag.parser.unified import Parser, MarkdownParseResult, parse_source_to_markdown
    assert Parser is not None
    assert MarkdownParseResult is not None
    assert parse_source_to_markdown is not None


def test_import_adapter():
    from app.rag.parser.adapter import DocumentProcessorFactory, BaseDocumentProcessor
    assert DocumentProcessorFactory is not None


def test_import_milvus_kb():
    from app.rag.backends.milvus_kb import MilvusKB, _run_milvus_query_io
    assert MilvusKB is not None


def test_import_chunk_repository():
    from app.rag.repositories.chunk_repository import KnowledgeChunkRepository
    repo = KnowledgeChunkRepository()
    assert hasattr(repo, "mark_graph_indexed")
    assert hasattr(repo, "reset_graph_state_by_kb_id")
    assert hasattr(repo, "list_unindexed_for_graph")


def test_import_config():
    from app.core.config import settings
    assert hasattr(settings, "MILVUS_QUERY_OFFLOAD_SEMAPHORE")
    assert hasattr(settings, "DEFAULT_SEARCH_MODE")
    assert hasattr(settings, "DEFAULT_VECTOR_WEIGHT")
    assert settings.DEFAULT_SEARCH_MODE == "hybrid"
