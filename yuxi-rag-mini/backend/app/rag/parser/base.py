from abc import ABC, abstractmethod
from pathlib import Path


class BaseParser(ABC):
    """Base class for file parsers."""

    @abstractmethod
    async def parse(self, file_path: str) -> str:
        """Parse file and return markdown text."""
        pass

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        pass
