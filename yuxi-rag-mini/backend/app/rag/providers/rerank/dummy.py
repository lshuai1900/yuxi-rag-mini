import random

from app.rag.providers.rerank.base import BaseReranker


class DummyReranker(BaseReranker):
    """Dummy reranker that returns original scores with slight random perturbation."""
    async def acompute_score(self, query: str, documents: list[str]) -> list[float]:
        return [random.uniform(0.5, 1.0) for _ in documents]
