"""Knowledge base QA service.

Pipeline:
    user question
      -> manager.aquery (reuse existing retrieval)
      -> build_context
      -> LLM (OpenAI-compatible chat completions)
      -> {answer, sources}
"""

from typing import Any

from app.core.logging import logger
from app.rag.manager import KnowledgeBaseManager
from app.rag.providers.llm import LLMProvider, create_llm_provider
from app.rag.schemas import CitationSchema

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。
你必须只根据给定的【参考资料】回答。
如果参考资料不足以回答，请直接说"知识库中没有找到足够信息回答这个问题"。
回答要简洁、准确。
涉及事实时，尽量在句子后标注引用编号，如 [1]、[2]。
不要编造来源。"""

NO_RESULTS_ANSWER = "知识库中没有找到足够信息回答这个问题。"


def _get_field(item: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict (by key) or an object (by attribute)."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def build_context(results: list[Any], max_chars: int = 12000) -> tuple[str, list[CitationSchema]]:
    """Build the LLM context string and citation list from retrieval results.

    Args:
        results: list of retrieval result items (dicts or objects).
        max_chars: hard cap on the total context length.

    Returns:
        (context_text, sources): context_text is the formatted references block,
        sources is the list of CitationSchema that were actually included.
    """
    if not results:
        return "", []

    context_parts: list[str] = []
    sources: list[CitationSchema] = []
    used_chars = 0
    # Dedicated citation counter: only increments for items we actually keep,
    # so the [1], [2], ... labels stay contiguous even when some results are
    # skipped (empty content) or truncated by the max_chars cap.
    citation_index = 0

    for item in results:
        content = _get_field(item, "content", "") or ""
        if not content.strip():
            # Skip empty content per requirement
            continue

        citation_index += 1
        block = (
            f"[{citation_index}] 文件: {_get_field(item, 'filename', '') or ''}\n"
            f"chunk_id: {_get_field(item, 'chunk_id', '') or ''}\n"
            f"chunk_index: {_get_field(item, 'chunk_index', 0)}\n"
            f"score: {float(_get_field(item, 'score', 0.0) or 0.0)}\n"
            f"内容:\n{content}\n"
            f"---"
        )

        if used_chars + len(block) > max_chars and context_parts:
            # Cap reached and we already have at least one block: stop.
            # Roll back the citation_index we just incremented since this
            # block is NOT included, so numbering stays consistent.
            citation_index -= 1
            logger.debug(
                f"build_context: stopping at {citation_index} blocks, cap={max_chars}"
            )
            break

        context_parts.append(block)
        used_chars += len(block)
        sources.append(
            CitationSchema(
                index=citation_index,
                chunk_id=str(_get_field(item, "chunk_id", "") or ""),
                file_id=str(_get_field(item, "file_id", "") or ""),
                filename=str(_get_field(item, "filename", "") or ""),
                chunk_index=int(_get_field(item, "chunk_index", 0) or 0),
                score=float(_get_field(item, "score", 0.0) or 0.0),
                content=content,
            )
        )

    context_text = "\n".join(context_parts)
    return context_text, sources


class KnowledgeQAService:
    """Glue between retrieval (manager.aquery) and the LLM provider."""

    def __init__(self, manager: KnowledgeBaseManager, llm_provider: LLMProvider | None = None):
        self.manager = manager
        self._llm_provider = llm_provider

    def _get_llm(self) -> LLMProvider:
        if self._llm_provider is None:
            self._llm_provider = create_llm_provider()
        return self._llm_provider

    async def answer(
        self,
        kb_id: str,
        query: str,
        search_mode: str = "hybrid",
        top_k: int = 6,
        similarity_threshold: float = 0.0,
        retrieval_config: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> dict:
        """Run retrieve -> build context -> LLM and return the chat result dict."""
        # 1. Retrieve
        query_kwargs: dict[str, Any] = {
            "search_mode": search_mode,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }
        if retrieval_config:
            query_kwargs["retrieval_config"] = retrieval_config

        results = await self.manager.aquery(query, kb_id, **query_kwargs)

        # 2. Build context + sources
        context_text, sources = build_context(results)

        # 3. No results -> refuse to answer
        if not sources:
            logger.info(f"QA: no retrieval results for kb={kb_id} query={query!r}")
            return {
                "query": query,
                "answer": NO_RESULTS_ANSWER,
                "sources": [],
                "search_mode": search_mode,
            }

        # 4. Call LLM
        llm = self._get_llm()
        if not llm.is_configured():
            raise RuntimeError(
                "LLM provider is not configured. Set LLM_BASE_URL and LLM_MODEL."
            )

        user_message = (
            f"【用户问题】\n{query}\n\n"
            f"【参考资料】\n{context_text}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        logger.info(
            f"QA: calling LLM model={llm.model} kb={kb_id} "
            f"sources={len(sources)} context_chars={len(context_text)}"
        )
        answer = await llm.achat(messages, temperature=temperature, max_tokens=max_tokens)

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "search_mode": search_mode,
        }
