import re
from typing import Any
from app.rag.chunking.text_chunker import count_tokens, _hard_split


def chunk_markdown(text: str, chunk_token_num: int = 512, overlapped_percent: int = 0) -> list[str]:
    """Chunk markdown text by headings first, then by token limit."""
    if not text or not text.strip():
        return []

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
        return [text.strip()] if text.strip() else []

    # Merge sections into chunks
    chunk_token_num = max(int(chunk_token_num or 0), 1)
    chunks = [""]
    token_nums = [0]

    for sec in sections:
        tnum = count_tokens(sec)
        if token_nums[-1] + tnum > chunk_token_num and chunks[-1].strip():
            chunks.append(sec)
            token_nums.append(tnum)
        else:
            chunks[-1] = (chunks[-1] + "\n\n" + sec).strip()
            token_nums[-1] += tnum

    # Hard split oversized chunks
    result = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        if count_tokens(chunk) <= chunk_token_num:
            result.append(chunk.strip())
        else:
            result.extend(_hard_split(chunk, chunk_token_num))

    return result
