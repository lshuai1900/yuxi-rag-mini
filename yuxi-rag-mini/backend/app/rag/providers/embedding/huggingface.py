from app.rag.providers.embedding.base import BaseEmbeddingProvider
from app.core.logging import logger


class HuggingFaceEmbedding(BaseEmbeddingProvider):
    """HuggingFace sentence-transformers embedding provider."""

    def __init__(self, model: str = "BAAI/bge-m3", dimension: int = 1024,
                 batch_size: int = 40, device: str | None = None, **kwargs):
        super().__init__(model=model, dimension=dimension, batch_size=batch_size, **kwargs)
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model, **kwargs)
        return self._model

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_documents([text])
        return results[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        model = self._get_model()
        embeddings = await asyncio.to_thread(model.encode, texts, show_progress_bar=False)
        return embeddings.tolist()
