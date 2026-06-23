import re
from typing import Any


def count_tokens(text: str) -> int:
    """Approximate token count."""
    if not text:
        return 0
    parts = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)
    return max(1, len(parts)) if text.strip() else 0


def chunk_text(text: str, chunk_token_num: int = 512, delimiter: str = "\n",
               overlapped_percent: int = 0) -> list[str]:
    """Simple text chunker based on delimiter and token limit."""
    if not text or not text.strip():
        return []

    chunk_token_num = max(int(chunk_token_num or 0), 1)
    overlap = max(0, min(int(overlapped_percent or 0), 99))

    # Split by delimiter
    if delimiter and delimiter in text:
        sections = [s.strip() for s in text.split(delimiter) if s.strip()]
    else:
        sections = [line.strip() for line in text.splitlines() if line.strip()]

    if not sections:
        return [text.strip()] if text.strip() else []

    # Merge sections into chunks respecting token limit
    chunks = [""]
    token_nums = [0]
    threshold = chunk_token_num * (100 - overlap) / 100.0

    for sec in sections:
        tnum = count_tokens(sec)
        if chunks[-1] == "" or token_nums[-1] > threshold:
            chunks.append("\n" + sec)
            token_nums.append(tnum)
        else:
            chunks[-1] += "\n" + sec
            token_nums[-1] += tnum

    # Hard split any chunks that exceed the limit
    result = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        if count_tokens(chunk) <= chunk_token_num:
            result.append(chunk.strip())
        else:
            result.extend(_hard_split(chunk, chunk_token_num))

    return result


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Split text that exceeds token limit."""
    token_iter = list(re.finditer(r"[A-Za-z0-9_]+|[一-鿿]", text))
    if not token_iter:
        return [text.strip()] if text.strip() else []

    spans = []
    start = 0
    index = 0
    while index < len(token_iter):
        next_index = min(index + max_tokens, len(token_iter))
        end = token_iter[next_index].start() if next_index < len(token_iter) else len(text)
        if text[start:end].strip():
            spans.append((start, end))
        start = end
        index = next_index

    return [text[s:e].strip() for s, e in spans if text[s:e].strip()]
