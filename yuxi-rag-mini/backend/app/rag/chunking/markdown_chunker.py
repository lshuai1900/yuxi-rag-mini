import re
from typing import Any
from app.rag.chunking.text_chunker import ChunkResult, count_tokens, _hard_split


def chunk_markdown(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    file_id: str = "",
    kb_id: str = "",
    filename: str = "",
    metadata: dict[str, Any] | None = None,
) -> list[ChunkResult]:
    """Chunk markdown text by headings first, then by token limit."""
    if not text or not text.strip():
        return []

    chunk_size = max(int(chunk_size), 1)
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    base_metadata = metadata or {}

    # Split by headings
    heading_pattern = re.compile(r'^(#{1,6}\s+.+)$', re.MULTILINE)
    sections = []
    last_end = 0

    for match in heading_pattern.finditer(text):
        if match.start() > last_end:
            pre_text = text[last_end:match.start()].strip()
            if pre_text:
                sections.append(pre_text)
        sections.append(match.group(1))
        last_end = match.end()

    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            sections.append(remaining)

    if not sections:
        sections = [text.strip()] if text.strip() else []

    # Merge sections into chunks
    raw_chunks: list[str] = []
    current_chunk = ""
    for sec in sections:
        if current_chunk and count_tokens(current_chunk + "\n\n" + sec) > chunk_size:
            raw_chunks.append(current_chunk.strip())
            if chunk_overlap > 0 and current_chunk.strip():
                from app.rag.chunking.text_chunker import _get_overlap_text
                overlap_text = _get_overlap_text(current_chunk, chunk_overlap)
                current_chunk = (overlap_text + "\n\n" + sec).strip() if overlap_text else sec
            else:
                current_chunk = sec
        else:
            current_chunk = (current_chunk + "\n\n" + sec).strip() if current_chunk else sec

    if current_chunk.strip():
        raw_chunks.append(current_chunk.strip())

    # Hard split oversized chunks
    final_chunks: list[str] = []
    for chunk in raw_chunks:
        if count_tokens(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(_hard_split(chunk, chunk_size))

    # Build ChunkResult list
    results = []
    for idx, content in enumerate(final_chunks):
        if not content.strip():
            continue
        chunk_id = f"{file_id}_chunk_{idx}" if file_id else f"chunk_{idx}"
        results.append(ChunkResult(
            content=content,
            chunk_index=idx,
            chunk_id=chunk_id,
            file_id=file_id,
            kb_id=kb_id,
            filename=filename,
            metadata={**base_metadata},
        ))

    # Ensure at least one chunk
    if not results and text.strip():
        chunk_id = f"{file_id}_chunk_0" if file_id else "chunk_0"
        results.append(ChunkResult(
            content=text.strip(),
            chunk_index=0,
            chunk_id=chunk_id,
            file_id=file_id,
            kb_id=kb_id,
            filename=filename,
            metadata={**base_metadata},
        ))

    return results
