from fastapi import APIRouter, HTTPException
from app.api.kb_routes import get_manager
from app.rag.schemas import QueryRequest, QueryResponse, SearchResultSchema
from app.rag.providers.rerank import create_reranker
from app.core.logging import logger

router = APIRouter(prefix="/api/kb/{kb_id}/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query_knowledge_base(kb_id: str, body: QueryRequest):
    manager = get_manager()
    try:
        # Check GraphRAG
        if body.enable_graphrag:
            return QueryResponse(
                query=body.query,
                search_mode=body.search_mode,
                results=[SearchResultSchema(
                    chunk_id="",
                    content="GraphRAG is reserved but not implemented yet.",
                    metadata={"warning": "GraphRAG is not implemented"},
                )],
            )

        results = await manager.aquery(
            body.query,
            kb_id,
            search_mode=body.search_mode,
            top_k=body.top_k,
            similarity_threshold=body.similarity_threshold,
        )

        # Apply rerank if requested
        if body.enable_rerank and results:
            try:
                reranker = create_reranker(provider="dummy")
                documents = [r.get("content", "") for r in results]
                scores = await reranker.acompute_score(body.query, documents)
                for i, r in enumerate(results):
                    if i < len(scores):
                        r["score"] = scores[i]
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
            except Exception as e:
                logger.warning(f"Rerank failed: {e}")

        search_results = []
        for r in results:
            search_results.append(SearchResultSchema(
                chunk_id=r.get("chunk_id", ""),
                file_id=r.get("file_id", ""),
                filename=r.get("filename", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            ))

        return QueryResponse(
            query=body.query,
            search_mode=body.search_mode,
            results=search_results,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
