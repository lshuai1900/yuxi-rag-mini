from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    @abstractmethod
    async def acompute_score(self, query: str, documents: list[str]) -> list[float]:
        pass

    async def aclose(self) -> None:
        pass
