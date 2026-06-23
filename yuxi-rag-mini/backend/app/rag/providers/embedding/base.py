from abc import ABC, abstractmethod
from typing import Any


class BaseEmbeddingProvider(ABC):
    def __init__(self, model: str, dimension: int = 1024, batch_size: int = 40, **kwargs):
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size

    @abstractmethod
    async def aencode(self, texts: list[str]) -> list[list[float]]:
        pass

    async def abatch_encode(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        batch_size = batch_size or self.batch_size
        data = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            res = await self.aencode(batch)
            data.extend(res)
        return data

    async def test_connection(self) -> tuple[bool, str]:
        try:
            embeddings = await self.aencode(["Hello world"])
            if self.dimension and len(embeddings[0]) != self.dimension:
                return False, f"Dimension mismatch: expected {self.dimension}, got {len(embeddings[0])}"
            return True, "OK"
        except Exception as e:
            return False, str(e)
