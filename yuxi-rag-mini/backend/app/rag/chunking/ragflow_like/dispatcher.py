from app.rag.chunking.ragflow_like.nlp import count_tokens
from app.rag.chunking.ragflow_like.presets import resolve_chunk_processing_params
from app.rag.chunking.ragflow_like.parsers.general import (
    chunk_markdown as general_chunk_markdown,
)


def chunk_markdown(
    markdown_content: str,
    file_id: str,
    filename: str,
    processing_params: dict | None = None,
) -> list[dict]:
    """Main entry point for ragflow_like chunking.

    Args:
        markdown_content: The markdown text to chunk.
        file_id: Identifier for the source file.
        filename: Name of the source file.
        processing_params: Optional dict with preset info and overrides.

    Returns:
        List of chunk record dicts with position info.
    """
    if not markdown_content or not markdown_content.strip():
        return []

    # Resolve processing params
    preset_id = "general"
    additional_params: dict | None = None

    if processing_params:
        preset_id = processing_params.get("chunk_preset_id", "general")
        # Extract additional params (everything except chunk_preset_id)
        additional_params = {
            k: v for k, v in processing_params.items() if k != "chunk_preset_id"
        }
        if not additional_params:
            additional_params = None

    params = resolve_chunk_processing_params(preset_id, additional_params)

    # Dispatch to appropriate parser
    text_chunks = _dispatch_markdown_parser(
        params["chunk_preset_id"],
        filename,
        markdown_content,
        parser_config=params,
    )

    if not text_chunks:
        return []

    # Build chunk records with position info
    return _build_chunk_records(text_chunks, file_id, filename, markdown_content)


def _dispatch_markdown_parser(
    preset_id: str,
    filename: str,
    markdown_content: str,
    parser_config: dict | None = None,
) -> list[str]:
    """Route to appropriate parser based on preset_id.

    Currently only supports "general" parser. Others fall back to general.

    Args:
        preset_id: The chunk preset identifier.
        filename: Name of the source file.
        markdown_content: The markdown text to chunk.
        parser_config: Optional parser configuration.

    Returns:
        List of chunk text strings.
    """
    # All presets currently use the general parser
    return general_chunk_markdown(markdown_content, parser_config=parser_config)


def _build_chunk_records(
    text_chunks: list[str],
    file_id: str,
    filename: str,
    source_text: str,
) -> list[dict]:
    """Convert text chunks to record dicts with position info.

    Args:
        text_chunks: List of chunk text strings.
        file_id: Identifier for the source file.
        filename: Name of the source file.
        source_text: The original source text (for calculating positions).

    Returns:
        List of dicts with: id, content, file_id, filename, chunk_index,
        chunk_id, start_char_pos, end_char_pos, start_token_pos, end_token_pos.
    """
    records: list[dict] = []
    search_offset = 0

    for idx, chunk_text in enumerate(text_chunks):
        if not chunk_text.strip():
            continue

        chunk_id = f"{file_id}_chunk_{idx}"

        # Find char positions in source text
        start_char_pos = source_text.find(chunk_text, search_offset)
        if start_char_pos == -1:
            # Fallback: try finding a stripped version
            start_char_pos = source_text.find(chunk_text.strip(), search_offset)
            if start_char_pos == -1:
                start_char_pos = search_offset

        end_char_pos = start_char_pos + len(chunk_text)
        search_offset = end_char_pos

        # Calculate token positions
        text_before = source_text[:start_char_pos]
        start_token_pos = count_tokens(text_before)
        end_token_pos = start_token_pos + count_tokens(chunk_text)

        records.append({
            "id": chunk_id,
            "content": chunk_text,
            "file_id": file_id,
            "filename": filename,
            "chunk_index": idx,
            "chunk_id": chunk_id,
            "start_char_pos": start_char_pos,
            "end_char_pos": end_char_pos,
            "start_token_pos": start_token_pos,
            "end_token_pos": end_token_pos,
        })

    return records
