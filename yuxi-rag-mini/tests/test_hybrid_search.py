"""Test hybrid search: vector, keyword, merge, score_detail, normalization."""
import os
import sys
import tempfile

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_test_dir = tempfile.mkdtemp(prefix="yuxi_rag_hybrid_")
os.environ["EMBEDDING_PROVIDER"] = "fake"
os.environ["EMBEDDING_DIMENSION"] = "128"
os.environ["DB_TYPE"] = "sqlite"
os.environ["SQLITE_PATH"] = os.path.join(_test_dir, "hybrid_test.db")
os.environ["STORAGE_TYPE"] = "local"
os.environ["LOCAL_STORAGE_PATH"] = os.path.join(_test_dir, "files")

from app.main import app
from app.rag.storage.database import init_db
from app.core.config import settings

settings.MILVUS_URI = os.path.join(_test_dir, "milvus.db")


@pytest_asyncio.fixture
async def client():
    await init_db()
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


async def _setup_kb_with_files(client: AsyncClient) -> str:
    """Helper: create KB, upload and index files."""
    kb_resp = await client.post("/api/kb", json={
        "name": "Hybrid Test KB",
        "description": "Test hybrid search",
    })
    assert kb_resp.status_code == 200
    kb_id = kb_resp.json()["kb_id"]

    # Upload a file with specific content
    content = "Python programming language for data science and AI machine learning"
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        with open(f.name, "rb") as upload_f:
            upload_resp = await client.post(
                f"/api/kb/{kb_id}/files/upload",
                files={"file": ("python_ai.txt", upload_f, "text/plain")},
            )
    os.unlink(f.name)
    assert upload_resp.status_code == 200
    file_id = upload_resp.json()["file_id"]

    # Index the file
    idx_resp = await client.post(f"/api/kb/{kb_id}/files/{file_id}/index")
    assert idx_resp.status_code == 200

    return kb_id


@pytest.mark.asyncio
async def test_vector_result_only(client):
    """Vector search should return results with score_detail."""
    kb_id = await _setup_kb_with_files(client)
    resp = await client.post(f"/api/kb/{kb_id}/query", json={
        "query": "Python AI",
        "search_mode": "vector",
        "top_k": 5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["search_mode"] == "vector"
    assert len(data["results"]) > 0
    result = data["results"][0]
    assert "score_detail" in result
    assert result["score_detail"]["source"] == "vector"
    assert result["score_detail"]["vector_score"] > 0


@pytest.mark.asyncio
async def test_keyword_result_only(client):
    """Keyword search should return results with score_detail."""
    kb_id = await _setup_kb_with_files(client)
    resp = await client.post(f"/api/kb/{kb_id}/query", json={
        "query": "Python programming",
        "search_mode": "keyword",
        "top_k": 5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["search_mode"] == "keyword"
    assert len(data["results"]) > 0
    result = data["results"][0]
    assert "score_detail" in result
    assert result["score_detail"]["source"] == "keyword"
    assert result["score_detail"]["keyword_score"] > 0


@pytest.mark.asyncio
async def test_hybrid_search_results(client):
    """Hybrid search should return results with combined score_detail."""
    kb_id = await _setup_kb_with_files(client)
    resp = await client.post(f"/api/kb/{kb_id}/query", json={
        "query": "Python AI machine learning",
        "search_mode": "hybrid",
        "top_k": 5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["search_mode"] == "hybrid"
    assert len(data["results"]) > 0

    for result in data["results"]:
        assert "score_detail" in result
        sd = result["score_detail"]
        assert sd["source"] == "hybrid"
        # At least one of vector_score or keyword_score should be > 0
        assert sd["vector_score"] > 0 or sd["keyword_score"] > 0
        assert sd["final_score"] > 0


@pytest.mark.asyncio
async def test_hybrid_score_detail_exists(client):
    """Verify score_detail has all required fields in hybrid search."""
    kb_id = await _setup_kb_with_files(client)
    resp = await client.post(f"/api/kb/{kb_id}/query", json={
        "query": "data science",
        "search_mode": "hybrid",
        "top_k": 5,
    })
    assert resp.status_code == 200
    data = resp.json()
    for result in data["results"]:
        sd = result["score_detail"]
        assert "vector_score" in sd
        assert "keyword_score" in sd
        assert "final_score" in sd
        assert "source" in sd


@pytest.mark.asyncio
async def test_hybrid_results_sorted_by_final_score(client):
    """Hybrid search results should be sorted by final_score descending."""
    kb_id = await _setup_kb_with_files(client)
    resp = await client.post(f"/api/kb/{kb_id}/query", json={
        "query": "Python AI",
        "search_mode": "hybrid",
        "top_k": 10,
    })
    assert resp.status_code == 200
    data = resp.json()
    if len(data["results"]) > 1:
        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)
