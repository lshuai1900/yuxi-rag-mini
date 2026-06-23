import hashlib
import struct
from app.rag.providers.embedding.base import BaseEmbeddingProvider


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Fake embedding provider for testing only.

    WARNING: This provider generates deterministic but meaningless vectors.
    It should ONLY be used for testing the pipeline, NOT for real RAG.
    Search results will be meaningless with this provider.
    """

    def __init__(self, dimension: int = 128, **kwargs):
        super().__init__(model="fake", dimension=dimension, batch_size=1000, **kwargs)

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_documents([text])
        return results[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._fake_embed(text) for text in texts]

    def _fake_embed(self, text: str) -> list[float]:
        """Generate a deterministic fake embedding based on text hash."""
        result = []
        for i in range(self.dimension):
            data = f"{text}:{i}".encode("utf-8")
            h = hashlib.md5(data).digest()
            val = struct.unpack("f", h[:4])[0]
            result.append(val)
        # Normalize to unit vector
        norm = sum(x * x for x in result) ** 0.5
        if norm > 0:
            result = [x / norm for x in result]
        return result
