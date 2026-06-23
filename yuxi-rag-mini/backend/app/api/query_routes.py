from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from app.api.kb_routes import get_manager
from app.rag.base import KnowledgeBase

router = APIRouter(prefix="/api/kb/{kb_id}/query", tags=["Query"])


class QueryRequest(BaseModel):
    query_text: str
    search_mode: Literal["vector", "keyword", "hybrid"] = "vector"
    top_k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    bm25_top_k: int = Field(default=50, ge=1, le=200)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)


@router.post("")
async def query_knowledge_base(kb_id: str, body: QueryRequest):
    manager = get_manager()
    try:
        results = await manager.aquery(
            body.query_text,
            kb_id,
            search_mode=body.search_mode,
            top_k=body.top_k,
            similarity_threshold=body.similarity_threshold,
            bm25_top_k=body.bm25_top_k,
            vector_weight=body.vector_weight,
            bm25_weight=body.bm25_weight,
        )
        kb_instance = await manager._get_kb_for_database(kb_id)
        return KnowledgeBase.build_search_output(kb_id, results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
