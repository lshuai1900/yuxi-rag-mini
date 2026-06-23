from fastapi import APIRouter, HTTPException
from app.api.kb_routes import get_manager
from app.rag.schemas import QueryRequest, QueryResponse, SearchResultSchema, ScoreDetail, RerankInfo, ErrorDetail
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

        # Build kwargs for aquery, including retrieval_config if provided
        query_kwargs = {
            "search_mode": body.search_mode,
            "top_k": body.top_k,
            "similarity_threshold": body.similarity_threshold,
        }
        if body.retrieval_config:
            query_kwargs["retrieval_config"] = body.retrieval_config

        results = await manager.aquery(
            body.query,
            kb_id,
            **query_kwargs,
        )

        # Apply rerank if requested
        rerank_info = None
        if body.enable_rerank and results:
            try:
                reranker = create_reranker(provider="dummy")
                documents = [r.get("content", "") for r in results]
                scores = await reranker.acompute_score(body.query, documents)
                for i, r in enumerate(results):
                    if i < len(scores):
                        r["score"] = scores[i]
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
                rerank_info = RerankInfo(reranked=True, reranker="dummy")
            except Exception as e:
                logger.warning(f"Rerank failed: {e}")
                rerank_info = RerankInfo(reranked=False, reranker="dummy")

        search_results = []
        for r in results:
            score_detail = None
            if "score_detail" in r:
                sd = r["score_detail"]
                score_detail = ScoreDetail(
                    vector_score=sd.get("vector_score", 0.0),
                    keyword_score=sd.get("keyword_score", 0.0),
                    final_score=sd.get("final_score", r.get("score", 0.0)),
                    source=sd.get("source", body.search_mode),
                )
            search_results.append(SearchResultSchema(
                chunk_id=r.get("chunk_id", ""),
                file_id=r.get("file_id", ""),
                filename=r.get("filename", ""),
                content=r.get("content", ""),
                chunk_index=r.get("chunk_index", 0),
                score=r.get("score", 0.0),
                score_detail=score_detail,
                metadata=r.get("metadata", {}),
            ))

        return QueryResponse(
            query=body.query,
            search_mode=body.search_mode,
            results=search_results,
            rerank=rerank_info,
        )
    except RuntimeError as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=503, detail=ErrorDetail(
            code="SERVICE_UNAVAILABLE",
            message=str(e),
            details={"kb_id": kb_id},
        ).model_dump())
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=400, detail=ErrorDetail(
            code="QUERY_FAILED",
            message=str(e),
            details={"kb_id": kb_id, "search_mode": body.search_mode},
        ).model_dump())
