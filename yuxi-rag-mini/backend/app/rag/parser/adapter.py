"""Optional document processor adapters.

These adapters wrap heavy external dependencies (MinerU, PaddleX, OCR)
that may not be installed. They are loaded lazily and don't break
startup if unavailable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.logging import logger


class DocumentProcessorException(Exception):
    """Base exception for document processors."""
    pass


class BaseDocumentProcessor(ABC):
    """Base class for optional document processors."""

    @abstractmethod
    async def process_file(self, file_path: str, **kwargs) -> str:
        """Process a file and return markdown content."""
        pass

    @abstractmethod
    def check_health(self) -> tuple[bool, str]:
        """Check if the processor is available and healthy."""
        pass

    def get_service_name(self) -> str:
        return self.__class__.__name__

    def get_supported_extensions(self) -> list[str]:
        return []


class DocumentProcessorFactory:
    """Factory for optional document processors with lazy loading."""

    _processors: dict[str, BaseDocumentProcessor] = {}
    _checked: set[str] = set()

    @classmethod
    def get_processor(cls, processor_type: str) -> BaseDocumentProcessor | None:
        """Get a processor by type, lazily initializing if needed."""
        if processor_type in cls._processors:
            return cls._processors[processor_type]

        if processor_type in cls._checked:
            return None

        cls._checked.add(processor_type)

        try:
            processor = cls._create_processor(processor_type)
            if processor:
                healthy, msg = processor.check_health()
                if healthy:
                    cls._processors[processor_type] = processor
                    logger.info(f"Document processor '{processor_type}' available")
                    return processor
                else:
                    logger.info(f"Document processor '{processor_type}' not healthy: {msg}")
            return None
        except Exception as e:
            logger.debug(f"Document processor '{processor_type}' not available: {e}")
            return None

    @classmethod
    def _create_processor(cls, processor_type: str) -> BaseDocumentProcessor | None:
        """Create a processor instance. Override or extend for custom processors."""
        # Stub: no heavy processors by default
        # Users can register custom processors
        return None

    @classmethod
    def register_processor(cls, processor_type: str, processor: BaseDocumentProcessor) -> None:
        """Register a custom processor."""
        cls._processors[processor_type] = processor

    @classmethod
    async def process_file(cls, file_path: str, processor_type: str = "", **kwargs) -> str | None:
        """Process a file using an optional processor."""
        processor = cls.get_processor(processor_type)
        if processor is None:
            return None
        return await processor.process_file(file_path, **kwargs)
