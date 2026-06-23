import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkResult:
    """Structured chunk result."""
    content: str
    chunk_index: int
    chunk_id: str = ""
    file_id: str = ""
    kb_id: str = ""
    filename: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def count_tokens(text: str) -> int:
    """Approximate token count (Chinese chars + English words)."""
    if not text:
        return 0
    parts = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)
    return max(1, len(parts)) if text.strip() else 0


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    file_id: str = "",
    kb_id: str = "",
    filename: str = "",
    metadata: dict[str, Any] | None = None,
) -> list[ChunkResult]:
    """Split text into chunks with overlap support."""
    if not text or not text.strip():
        return []

    chunk_size = max(int(chunk_size), 1)
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    base_metadata = metadata or {}

    # Split by paragraphs/lines first
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    # Merge lines into chunks respecting chunk_size
    raw_chunks: list[str] = []
    current_chunk = ""
    for line in lines:
        # If adding this line exceeds chunk_size and current_chunk is not empty
        if current_chunk and count_tokens(current_chunk + "\n" + line) > chunk_size:
            raw_chunks.append(current_chunk.strip())
            # Handle overlap: keep last few tokens worth of text
            if chunk_overlap > 0 and current_chunk.strip():
                overlap_text = _get_overlap_text(current_chunk, chunk_overlap)
                current_chunk = overlap_text + "\n" + line if overlap_text else line
            else:
                current_chunk = line
        else:
            current_chunk = (current_chunk + "\n" + line).strip() if current_chunk else line

    if current_chunk.strip():
        raw_chunks.append(current_chunk.strip())

    # Hard split any chunks that still exceed the limit
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
        chunk_metadata = {**base_metadata}
        results.append(ChunkResult(
            content=content,
            chunk_index=idx,
            chunk_id=chunk_id,
            file_id=file_id,
            kb_id=kb_id,
            filename=filename,
            metadata=chunk_metadata,
        ))

    # Ensure at least one chunk for short text
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


def _get_overlap_text(text: str, overlap_tokens: int) -> str:
    """Get the last N tokens worth of text for overlap."""
    token_iter = list(re.finditer(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text))
    if len(token_iter) <= overlap_tokens:
        return text
    start_token = token_iter[-overlap_tokens]
    return text[start_token.start():]


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Split text that exceeds token limit by character positions."""
    token_iter = list(re.finditer(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text))
    if not token_iter:
        return [text.strip()] if text.strip() else []

    spans = []
    start_idx = 0
    while start_idx < len(token_iter):
        end_idx = min(start_idx + max_tokens, len(token_iter))
        char_start = token_iter[start_idx].start()
        char_end = token_iter[end_idx - 1].end() if end_idx > 0 else len(text)
        chunk = text[char_start:char_end].strip()
        if chunk:
            spans.append(chunk)
        start_idx = end_idx

    return spans
