from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParseResult:
    """Structured result from file parsing."""
    text: str
    filename: str = ""
    file_type: str = ""
    pages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    """Base class for file parsers."""

    @abstractmethod
    async def parse(self, file_path: str, filename: str = "") -> ParseResult:
        """Parse file and return structured result."""
        pass

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        pass
