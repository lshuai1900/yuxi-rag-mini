import httpx

from app.rag.providers.embedding.base import BaseEmbeddingProvider


class OllamaEmbedding(BaseEmbeddingProvider):
    """Ollama embedding provider - uses OpenAI-compatible endpoint from Ollama."""
    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434/v1",
                 dimension: int = 1024, batch_size: int = 40, **kwargs):
        super().__init__(model=model, dimension=dimension, batch_size=batch_size, **kwargs)
        self.base_url = base_url.rstrip("/") + "/embeddings"
        self.headers = {"Content-Type": "application/json"}

    async def aencode(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            if "data" not in result:
                raise ValueError(f"Invalid embedding response: {result}")
            sorted_data = sorted(result["data"], key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in sorted_data]
