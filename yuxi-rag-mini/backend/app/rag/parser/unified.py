import os
from dataclasses import dataclass, field
from typing import Any

from app.rag.parser.factory import get_parser
from app.core.logging import logger


@dataclass
class MarkdownParseResult:
    """Result of parsing a file to markdown."""
    markdown: str
    file_ext: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)


class Parser:
    """Unified parser that converts files to markdown."""

    @classmethod
    async def aparse(cls, source: str, params: dict | None = None) -> str:
        """Parse a file and return markdown string.

        Args:
            source: File path to parse
            params: Optional parsing parameters

        Returns:
            Markdown string of the parsed content
        """
        params = params or {}
        filename = params.get("filename", os.path.basename(source))

        try:
            parser = get_parser(source)
            result = await parser.parse(source, filename=filename)
        except ValueError:
            # Unsupported file type - try reading as plain text
            logger.warning(f"No parser for {source}, reading as plain text")
            try:
                with open(source, encoding="utf-8", errors="replace") as f:
                    return f.read()
            except Exception:
                raise ValueError(f"Cannot parse file: {source}")

        # Convert ParseResult to markdown
        markdown = result.text
        if not markdown or not markdown.strip():
            raise ValueError(f"No extractable content from {filename}")

        return markdown


async def parse_source_to_markdown(source: str, params: dict | None = None) -> MarkdownParseResult:
    """Parse a file source to markdown.

    Args:
        source: File path to parse
        params: Optional parsing parameters

    Returns:
        MarkdownParseResult with markdown content and metadata
    """
    params = params or {}
    file_ext = os.path.splitext(source)[1].lower() if source else ""

    markdown = await Parser.aparse(source, params)

    artifacts = {}
    if "filename" in params:
        artifacts["filename"] = params["filename"]

    return MarkdownParseResult(
        markdown=markdown,
        file_ext=file_ext,
        artifacts=artifacts,
    )
