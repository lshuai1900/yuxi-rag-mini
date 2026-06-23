from app.rag.providers.rerank.base import BaseReranker
from app.rag.providers.rerank.dummy import DummyReranker
from app.rag.providers.rerank.api_reranker import APIReranker


def create_reranker(provider: str = "dummy", **kwargs) -> BaseReranker:
    if provider == "dummy":
        return DummyReranker()
    elif provider == "api":
        return APIReranker(**kwargs)
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")
