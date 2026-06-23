import re

from app.rag.chunking.ragflow_like.nlp import (
    count_tokens,
    naive_merge,
    hard_split_by_token_limit,
)


def chunk_markdown(
    markdown_content: str,
    parser_config: dict | None = None,
) -> list[str]:
    """Chunk markdown text by headings, then merge into token-limited chunks.

    Args:
        markdown_content: The markdown text to chunk.
        parser_config: Optional config with keys like chunk_token_num,
            delimiter, overlapped_percent, hard_limit_token_num.

    Returns:
        List of chunk text strings.
    """
    if not markdown_content or not markdown_content.strip():
        return []

    config = parser_config or {}
    chunk_token_num = config.get("chunk_token_num", 512)
    delimiter = config.get("delimiter", "\n")
    overlapped_percent = config.get("overlapped_percent", 0.0)
    hard_limit_token_num = config.get("hard_limit_token_num", 0)

    sections = _iter_sections(markdown_content)
    if not sections:
        return [markdown_content.strip()] if markdown_content.strip() else []

    chunks = naive_merge(
        sections,
        chunk_token_num=chunk_token_num,
        delimiter=delimiter,
        overlapped_percent=overlapped_percent,
    )

    chunks = _ensure_chunk_token_limit(chunks, hard_limit=hard_limit_token_num)

    return chunks


def _iter_sections(markdown_content: str) -> list[str]:
    """Split markdown by headings into sections.

    Each heading and its content becomes a section.

    Args:
        markdown_content: The markdown text to split.

    Returns:
        List of section strings.
    """
    if not markdown_content or not markdown_content.strip():
        return []

    heading_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
    sections: list[str] = []
    last_end = 0

    for match in heading_pattern.finditer(markdown_content):
        if match.start() > last_end:
            pre_text = markdown_content[last_end:match.start()].strip()
            if pre_text:
                sections.append(pre_text)
        sections.append(match.group(1))
        last_end = match.end()

    if last_end < len(markdown_content):
        remaining = markdown_content[last_end:].strip()
        if remaining:
            sections.append(remaining)

    if not sections:
        sections = [markdown_content.strip()] if markdown_content.strip() else []

    return sections


def _ensure_chunk_token_limit(
    chunks: list[str],
    hard_limit: int = 0,
) -> list[str]:
    """Enforce max token limit on chunks.

    If hard_limit > 0, split any chunk that exceeds it using hard_split_by_token_limit.
    Otherwise return chunks as-is.

    Args:
        chunks: List of chunk strings.
        hard_limit: Maximum tokens per chunk. 0 means no hard limit.

    Returns:
        List of chunk strings respecting the hard limit.
    """
    if hard_limit <= 0:
        return chunks

    result: list[str] = []
    for chunk in chunks:
        if count_tokens(chunk) > hard_limit:
            result.extend(hard_split_by_token_limit(chunk, hard_limit, hard_limit))
        else:
            result.append(chunk)

    return result
