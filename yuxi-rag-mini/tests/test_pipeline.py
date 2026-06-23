"""
Yuxi RAG Mini - Full Pipeline Test

Tests the complete RAG pipeline from knowledge base creation to query.
Uses FakeEmbeddingProvider and Milvus Lite for testing without external dependencies.
"""
import asyncio
import os
import sys
import tempfile

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set test environment before importing app
_test_dir = tempfile.mkdtemp(prefix="yuxi_rag_test_")
os.environ["EMBEDDING_PROVIDER"] = "fake"
os.environ["EMBEDDING_DIMENSION"] = "128"
# Don't set MILVUS_URI env var - pymilvus singleton rejects non-http URIs
# Instead, we'll override settings.MILVUS_URI after import
os.environ["DB_TYPE"] = "sqlite"
os.environ["SQLITE_PATH"] = os.path.join(_test_dir, "yuxi_rag.db")
os.environ["STORAGE_TYPE"] = "local"
os.environ["LOCAL_STORAGE_PATH"] = os.path.join(_test_dir, "files")

from app.main import app
from app.rag.storage.database import init_db
from app.core.config import settings

# Override Milvus Uri to use Milvus Lite (local file) for testing
settings.MILVUS_URI = os.path.join(_test_dir, "milvus.db")


@pytest_asyncio.fixture
async def client():
    """Create async test client with initialized manager."""
    await init_db()

    # Initialize KnowledgeBaseManager (normally done in lifespan)
    from app.rag.factory import KnowledgeBaseFactory
    from app.rag.backends.milvus_kb import MilvusKB
    from app.rag.manager import KnowledgeBaseManager
    from app.api import kb_routes

    KnowledgeBaseFactory.register(MilvusKB)
    manager = KnowledgeBaseManager(work_dir=os.path.join(_test_dir, "kb_work"))
    await manager.initialize()
    kb_routes.set_manager(manager)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client):
    """Test 1: Health check endpoint."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "yuxi-rag-mini"


@pytest.mark.asyncio
async def test_create_knowledge_base(client):
    """Test 2: Create a knowledge base."""
    response = await client.post("/api/kb", json={
        "name": "Test KB",
        "description": "Test knowledge base for pipeline",
    })
    assert response.status_code == 200
    data = response.json()
    assert "kb_id" in data
    assert data["name"] == "Test KB"
    assert data["kb_type"] == "milvus"
    return data["kb_id"]


@pytest.mark.asyncio
async def test_list_knowledge_bases(client):
    """Test 3: List knowledge bases."""
    response = await client.get("/api/kb")
    assert response.status_code == 200
    data = response.json()
    assert "databases" in data


@pytest.mark.asyncio
async def test_full_rag_pipeline(client):
    """Test 4-12: Full RAG pipeline test."""
    # Create KB
    kb_resp = await client.post("/api/kb", json={
        "name": "Pipeline Test KB",
        "description": "Full pipeline test",
    })
    assert kb_resp.status_code == 200
    kb_id = kb_resp.json()["kb_id"]

    # Create test files
    md_content = """# AI Overview
Artificial intelligence is the simulation of human intelligence.
Machine learning is a subset of AI.
Deep learning uses neural networks with multiple layers.
"""
    txt_content = "Python is a programming language. It is widely used in data science and AI."

    # Upload MD
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(md_content)
        f.flush()
        with open(f.name, "rb") as upload_f:
            md_resp = await client.post(
                f"/api/kb/{kb_id}/files/upload",
                files={"file": ("ai_overview.md", upload_f, "text/markdown")},
            )
    assert md_resp.status_code == 200
    md_file_id = md_resp.json()["file_id"]
    os.unlink(f.name)

    # Upload TXT
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write(txt_content)
        f.flush()
        with open(f.name, "rb") as upload_f:
            txt_resp = await client.post(
                f"/api/kb/{kb_id}/files/upload",
                files={"file": ("python.txt", upload_f, "text/plain")},
            )
    assert txt_resp.status_code == 200
    txt_file_id = txt_resp.json()["file_id"]
    os.unlink(f.name)

    # Index MD
    idx_md_resp = await client.post(f"/api/kb/{kb_id}/files/{md_file_id}/index")
    assert idx_md_resp.status_code == 200
    idx_data = idx_md_resp.json()
    assert idx_data["status"] == "indexed"
    assert idx_data["chunk_count"] >= 1

    # Index TXT
    idx_txt_resp = await client.post(f"/api/kb/{kb_id}/files/{txt_file_id}/index")
    assert idx_txt_resp.status_code == 200
    idx_data = idx_txt_resp.json()
    assert idx_data["status"] == "indexed"
    assert idx_data["chunk_count"] >= 1

    # Vector search
    vec_resp = await client.post(f"/api/kb/{kb_id}/query", json={
        "query": "What is deep learning?",
        "search_mode": "vector",
        "top_k": 5,
    })
    assert vec_resp.status_code == 200
    vec_data = vec_resp.json()
    assert vec_data["search_mode"] == "vector"
    assert len(vec_data["results"]) > 0
    # Verify score_detail exists
    assert vec_data["results"][0]["score_detail"] is not None
    assert vec_data["results"][0]["score_detail"]["source"] == "vector"

    # Keyword search
    kw_resp = await client.post(f"/api/kb/{kb_id}/query", json={
        "query": "Python programming",
        "search_mode": "keyword",
        "top_k": 5,
    })
    assert kw_resp.status_code == 200
    kw_data = kw_resp.json()
    assert kw_data["search_mode"] == "keyword"
    assert len(kw_data["results"]) > 0
    # Verify score_detail exists
    assert kw_data["results"][0]["score_detail"] is not None
    assert kw_data["results"][0]["score_detail"]["source"] == "keyword"

    # Hybrid search
    hyb_resp = await client.post(f"/api/kb/{kb_id}/query", json={
        "query": "machine learning AI",
        "search_mode": "hybrid",
        "top_k": 5,
    })
    assert hyb_resp.status_code == 200
    hyb_data = hyb_resp.json()
    assert hyb_data["search_mode"] == "hybrid"
    assert len(hyb_data["results"]) > 0
    # Verify score_detail with hybrid source
    assert hyb_data["results"][0]["score_detail"] is not None
    assert hyb_data["results"][0]["score_detail"]["source"] == "hybrid"

    # GraphRAG placeholder
    graph_resp = await client.post(f"/api/kb/{kb_id}/query", json={
        "query": "test",
        "search_mode": "vector",
        "enable_graphrag": True,
    })
    assert graph_resp.status_code == 200
    graph_data = graph_resp.json()
    assert "not implemented" in graph_data["results"][0]["content"].lower()


@pytest.mark.asyncio
async def test_no_yuxi_imports():
    """Test: No yuxi.* imports in the codebase."""
    import ast
    backend_dir = os.path.join(os.path.dirname(__file__), "..", "backend", "app")
    errors = []
    for root, dirs, files in os.walk(backend_dir):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path) as fh:
                    try:
                        tree = ast.parse(fh.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("yuxi"):
                                errors.append(f"{path} imports from yuxi: {node.module}")
                    except SyntaxError:
                        pass
    assert len(errors) == 0, f"Found yuxi.* imports: {errors}"


@pytest.mark.asyncio
async def test_backend_import():
    """Test: backend/app can be fully imported."""
    import importlib
    try:
        importlib.import_module("app.main")
        importlib.import_module("app.rag.schemas")
        importlib.import_module("app.rag.base")
        importlib.import_module("app.rag.backends.milvus_kb")
        importlib.import_module("app.rag.parser.factory")
        importlib.import_module("app.rag.chunking.text_chunker")
        importlib.import_module("app.rag.providers.embedding.fake")
        importlib.import_module("app.rag.providers.rerank.dummy")
        importlib.import_module("app.core.config")
    except ImportError as e:
        pytest.fail(f"Failed to import: {e}")
