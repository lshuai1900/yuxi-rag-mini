"""GraphRAG interfaces - placeholder for future implementation.

This module defines the interfaces for GraphRAG functionality.
The first version only provides interfaces, no Neo4j implementation.
"""

from abc import ABC, abstractmethod
from typing import Any


class GraphStore(ABC):
    """Interface for graph storage operations."""

    @abstractmethod
    async def create_graph(self, kb_id: str) -> None:
        """Create a graph for the given knowledge base."""
        pass

    @abstractmethod
    async def add_entities(self, kb_id: str, entities: list[dict]) -> None:
        """Add entities to the graph."""
        pass

    @abstractmethod
    async def add_triples(self, kb_id: str, triples: list[dict]) -> None:
        """Add triples to the graph."""
        pass

    @abstractmethod
    async def query_graph(self, kb_id: str, query: str, top_k: int = 10) -> list[dict]:
        """Query the graph and return relevant chunks."""
        pass

    @abstractmethod
    async def delete_graph(self, kb_id: str) -> None:
        """Delete the graph for the given knowledge base."""
        pass


class GraphExtractor(ABC):
    """Interface for extracting graph data from text."""

    @abstractmethod
    async def extract(self, text: str) -> dict[str, list[dict]]:
        """Extract entities and triples from text."""
        pass
