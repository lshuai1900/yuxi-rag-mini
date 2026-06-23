import re


def count_tokens(text: str) -> int:
    """Approximate token count (Chinese chars + English words)."""
    if not text:
        return 0
    parts = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)
    return max(1, len(parts)) if text.strip() else 0


def naive_merge(
    sections: list[str],
    chunk_token_num: int,
    delimiter: str = "\n",
    overlapped_percent: float = 0.0,
) -> list[str]:
    """Merge sections into chunks where each chunk has approximately chunk_token_num tokens.

    Args:
        sections: List of text sections to merge.
        chunk_token_num: Target number of tokens per chunk.
        delimiter: String used to join sections.
        overlapped_percent: Fraction of tokens to overlap between chunks (0.0 to 1.0).

    Returns:
        List of merged chunk strings.
    """
    if not sections:
        return []

    overlapped_percent = max(0.0, min(1.0, overlapped_percent))
    chunks: list[str] = []
    current_chunk = ""

    for section in sections:
        if not section.strip():
            continue

        candidate = current_chunk + delimiter + section if current_chunk else section
        token_count = count_tokens(candidate)

        if token_count > chunk_token_num and current_chunk:
            chunks.append(current_chunk.strip())

            # Handle overlap
            if overlapped_percent > 0 and current_chunk.strip():
                overlap_token_count = max(1, int(count_tokens(current_chunk) * overlapped_percent))
                overlap_text = _get_overlap_text(current_chunk, overlap_token_count)
                current_chunk = overlap_text + delimiter + section if overlap_text else section
            else:
                current_chunk = section
        else:
            current_chunk = candidate

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _get_overlap_text(text: str, overlap_tokens: int) -> str:
    """Get the last N tokens worth of text for overlap."""
    token_iter = list(re.finditer(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text))
    if len(token_iter) <= overlap_tokens:
        return text
    start_token = token_iter[-overlap_tokens]
    return text[start_token.start():]


def hard_split_by_token_limit(
    text: str,
    chunk_token_num: int,
    hard_limit_token_num: int = 0,
) -> list[str]:
    """Fallback protection: split text if it exceeds hard_limit_token_num.

    Args:
        text: The text to potentially split.
        chunk_token_num: Target tokens per chunk (used when hard_limit is 0).
        hard_limit_token_num: If > 0 and text exceeds this, split by token positions.

    Returns:
        List of text chunks, or [text] if no splitting needed.
    """
    if not text or not text.strip():
        return []

    limit = hard_limit_token_num if hard_limit_token_num > 0 else 0
    if limit <= 0:
        return [text]

    token_count = count_tokens(text)
    if token_count <= limit:
        return [text]

    # Split by token positions
    token_iter = list(re.finditer(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text))
    if not token_iter:
        return [text.strip()] if text.strip() else []

    spans: list[str] = []
    start_idx = 0
    while start_idx < len(token_iter):
        end_idx = min(start_idx + limit, len(token_iter))
        char_start = token_iter[start_idx].start()
        char_end = token_iter[end_idx - 1].end() if end_idx > 0 else len(text)
        chunk = text[char_start:char_end].strip()
        if chunk:
            spans.append(chunk)
        start_idx = end_idx

    return spans


def bullets_category(sections: list[str]) -> int:
    """Detect document bullet pattern depth.

    Returns the max nesting depth of bullet patterns.
    Detects patterns like "1.", "1.1", "1.1.1", "-", "*", etc.
    """
    if not sections:
        return 0

    max_depth = 0
    bullet_pattern = re.compile(r"^(\s*)((?:\d+\.)*\d+\.?\s|[-*•]\s)")

    for section in sections:
        for line in section.splitlines():
            m = bullet_pattern.match(line)
            if m:
                indent = len(m.group(1))
                # Count dot-separated levels like "1.1.1"
                numbering = re.match(r"^((?:\d+\.)*\d+)", line.strip())
                if numbering:
                    depth = numbering.group(1).count(".") + 1
                else:
                    depth = 1 + indent // 2
                max_depth = max(max_depth, depth)

    return max_depth


def tree_merge(bull: int, sections: list[str], depth: int = 0) -> list[str]:
    """Hierarchical merge by bullet structure.

    Args:
        bull: Bullet nesting depth.
        sections: List of text sections.
        depth: Current recursion depth.

    Returns:
        List of merged sections.
    """
    if not sections:
        return []

    if bull <= 0:
        return sections

    # Group sections by their top-level bullet prefix
    groups: list[list[str]] = []
    current_group: list[str] = []

    for section in sections:
        if _is_top_level_bullet(section, depth):
            if current_group:
                groups.append(current_group)
            current_group = [section]
        else:
            current_group.append(section)

    if current_group:
        groups.append(current_group)

    # Merge each group into a single chunk
    result: list[str] = []
    for group in groups:
        merged = "\n".join(group)
        if merged.strip():
            result.append(merged.strip())

    return result


def _is_top_level_bullet(section: str, depth: int) -> bool:
    """Check if a section starts with a top-level bullet at the given depth."""
    if not section.strip():
        return False
    first_line = section.splitlines()[0].strip()
    # Match numbered patterns like "1.", "1.1", etc.
    numbering = re.match(r"^((?:\d+\.){})".format("{" + str(depth) + "}"), first_line)
    if numbering:
        return True
    # Match simple bullet characters
    if depth == 0 and re.match(r"^[-*•]\s", first_line):
        return True
    return False


def hierarchical_merge(bull: int, sections: list[str], depth: int = 0) -> list[str]:
    """Another merge strategy using hierarchical structure.

    Args:
        bull: Bullet nesting depth.
        sections: List of text sections.
        depth: Current recursion depth.

    Returns:
        List of merged sections.
    """
    if not sections:
        return []

    if bull <= 0:
        return sections

    # Simple hierarchical merge: group consecutive sections with same-level bullets
    result: list[str] = []
    current_chunk = ""

    for section in sections:
        if not section.strip():
            continue
        candidate = current_chunk + "\n" + section if current_chunk else section
        # If this section starts a new top-level bullet and we have accumulated content
        if _is_top_level_bullet(section, depth) and current_chunk:
            result.append(current_chunk.strip())
            current_chunk = section
        else:
            current_chunk = candidate

    if current_chunk.strip():
        result.append(current_chunk.strip())

    return result


def is_english(texts: list[str]) -> bool:
    """Detect if text is primarily English.

    Returns True if English characters outnumber Chinese characters.
    """
    if not texts:
        return True

    english_count = 0
    chinese_count = 0
    for text in texts:
        english_count += len(re.findall(r"[A-Za-z]", text))
        chinese_count += len(re.findall(r"[\u4e00-\u9fff]", text))

    return english_count >= chinese_count


def not_bullet(line: str) -> bool:
    """Check if line is NOT a bullet point."""
    if not line or not line.strip():
        return True
    stripped = line.strip()
    # Check for numbered bullets: "1.", "1.1", "1.1.1", etc.
    if re.match(r"^\d+(\.\d+)*\.?\s", stripped):
        return False
    # Check for symbol bullets: "-", "*", "•"
    if re.match(r"^[-*•]\s", stripped):
        return False
    return True


def is_probable_heading_line(line: str) -> bool:
    """Detect if line looks like a heading.

    Checks for markdown heading syntax or short lines that look like titles.
    """
    if not line or not line.strip():
        return False
    stripped = line.strip()

    # Markdown heading
    if re.match(r"^#{1,6}\s+", stripped):
        return True

    # Short line (likely a heading) that doesn't end with punctuation
    # and is not a bullet point
    if len(stripped) < 80 and not_bullet(stripped):
        if not re.search(r"[。？！，、；：,.!?;:]$", stripped):
            return True

    return False
