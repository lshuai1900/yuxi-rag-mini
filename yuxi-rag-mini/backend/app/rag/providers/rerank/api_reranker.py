import httpx

from app.rag.providers.rerank.base import BaseReranker
from app.core.logging import logger


class APIReranker(BaseReranker):
    """OpenAI-compatible reranker API."""
    def __init__(self, model: str, base_url: str, api_key: str, **kwargs):
        self.model = model
        self.base_url = base_url.rstrip("/") + "/rerank"
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def acompute_score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        payload = {"model": self.model, "query": query, "documents": documents}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            result = response.json()
        results = sorted(result.get("results", []), key=lambda x: x.get("index", 0))
        return [float(r.get("relevance_score", 0.0)) for r in results]
