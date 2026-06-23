from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""

    def __init__(self, model: str, dimension: int = 1024, batch_size: int = 40, **kwargs):
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts."""
        pass

    async def test_connection(self) -> tuple[bool, str]:
        """Test if the embedding provider is available."""
        try:
            embedding = await self.embed_query("Hello world")
            if self.dimension and len(embedding) != self.dimension:
                return False, f"Dimension mismatch: expected {self.dimension}, got {len(embedding)}"
            return True, "OK"
        except Exception as e:
            return False, str(e)
