from fastapi import APIRouter, HTTPException

from app.api.kb_routes import get_manager
from app.core.logging import logger
from app.rag.providers.llm import create_llm_provider
from app.rag.qa_service import KnowledgeQAService
from app.rag.schemas import ChatRequest, ChatResponse, CitationSchema, ErrorDetail

router = APIRouter(prefix="/api/kb/{kb_id}/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat_knowledge_base(kb_id: str, body: ChatRequest):
    """Knowledge base QA endpoint.

    Pipeline: retrieve -> build context -> LLM -> answer + sources.

    The LLM provider is created lazily (only when needed) and the
    configuration check is performed inside the service, so an empty
    retrieval result will short-circuit to the "no information" answer
    without ever touching the LLM (and without raising 503).
    """
    manager = get_manager()
    try:
        # Build the provider lazily here; the service will only contact it
        # when retrieval actually returned usable context.
        llm_provider = create_llm_provider()
        service = KnowledgeQAService(manager, llm_provider=llm_provider)

        result = await service.answer(
            kb_id=kb_id,
            query=body.query,
            search_mode=body.search_mode,
            top_k=body.top_k,
            similarity_threshold=body.similarity_threshold,
            retrieval_config=body.retrieval_config,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )

        sources = [
            CitationSchema(**s) if isinstance(s, dict) else s
            for s in result.get("sources", [])
        ]
        return ChatResponse(
            query=result["query"],
            answer=result["answer"],
            sources=sources,
            search_mode=result.get("search_mode", body.search_mode),
        )
    except HTTPException:
        # Re-raise already-built HTTP errors untouched
        raise
    except RuntimeError as e:
        # LLM not configured / call failed -> 503
        logger.error(f"Chat failed (runtime): {e}")
        raise HTTPException(status_code=503, detail=ErrorDetail(
            code="LLM_SERVICE_UNAVAILABLE",
            message=str(e),
            details={"kb_id": kb_id},
        ).model_dump())
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=400, detail=ErrorDetail(
            code="CHAT_FAILED",
            message=str(e),
            details={"kb_id": kb_id, "search_mode": body.search_mode},
        ).model_dump())

