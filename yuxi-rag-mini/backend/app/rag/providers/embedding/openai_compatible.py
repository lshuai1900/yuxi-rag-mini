import asyncio
import os

import httpx

from app.rag.providers.embedding.base import BaseEmbeddingProvider
from app.core.logging import logger


class OpenAICompatibleEmbedding(BaseEmbeddingProvider):
    def __init__(self, model: str = "bge-m3", base_url: str = "", api_key: str = "",
                 dimension: int = 1024, batch_size: int = 40, **kwargs):
        super().__init__(model=model, dimension=dimension, batch_size=batch_size, **kwargs)
        self.base_url = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def aencode(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            if "data" not in result:
                raise ValueError(f"Invalid embedding response: {result}")
            sorted_data = sorted(result["data"], key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in sorted_data]
