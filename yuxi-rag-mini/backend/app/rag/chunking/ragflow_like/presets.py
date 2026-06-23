VALID_PRESET_IDS = {"general", "qa", "book", "laws", "semantic", "separator"}

_PRESET_DEFAULTS: dict[str, dict] = {
    "general": {
        "chunk_token_num": 512,
        "delimiter": "\n",
        "overlapped_percent": 0.0,
        "hard_limit_token_num": 0,
    },
    "qa": {
        "chunk_token_num": 512,
        "delimiter": "\n",
        "overlapped_percent": 0.0,
        "hard_limit_token_num": 0,
    },
    "book": {
        "chunk_token_num": 1024,
        "delimiter": "\n",
        "overlapped_percent": 0.0,
        "hard_limit_token_num": 0,
    },
    "laws": {
        "chunk_token_num": 2048,
        "delimiter": "\n",
        "overlapped_percent": 0.0,
        "hard_limit_token_num": 0,
    },
    "semantic": {
        "chunk_token_num": 512,
        "delimiter": "\n",
        "overlapped_percent": 0.0,
        "hard_limit_token_num": 0,
    },
    "separator": {
        "chunk_token_num": 512,
        "delimiter": "\n\n",
        "overlapped_percent": 0.0,
        "hard_limit_token_num": 0,
    },
}

_EXTERNAL_NAME_MAP: dict[str, str] = {
    "default": "general",
    "manual": "general",
    "paper": "book",
    "resume": "general",
    "table": "general",
    "picture": "general",
    "one": "general",
}


def normalize_chunk_preset_id(preset_id: str) -> str:
    """Normalize and validate preset ID, defaulting to 'general'."""
    if not preset_id:
        return "general"
    normalized = preset_id.strip().lower()
    if normalized in VALID_PRESET_IDS:
        return normalized
    return "general"


def map_to_internal_parser_id(preset_id: str) -> str:
    """Map external names to internal parser IDs."""
    normalized = normalize_chunk_preset_id(preset_id)
    return _EXTERNAL_NAME_MAP.get(normalized, normalized)


def resolve_chunk_processing_params(
    preset_id: str = "general",
    additional_params: dict | None = None,
) -> dict:
    """Resolve processing params from preset.

    Args:
        preset_id: Preset identifier string.
        additional_params: Optional dict to override preset defaults.

    Returns:
        Dict with keys: chunk_preset_id, chunk_token_num, delimiter,
        overlapped_percent, hard_limit_token_num.
    """
    normalized = normalize_chunk_preset_id(preset_id)
    defaults = _PRESET_DEFAULTS.get(normalized, _PRESET_DEFAULTS["general"]).copy()

    result = {
        "chunk_preset_id": normalized,
        "chunk_token_num": defaults["chunk_token_num"],
        "delimiter": defaults["delimiter"],
        "overlapped_percent": defaults["overlapped_percent"],
        "hard_limit_token_num": defaults["hard_limit_token_num"],
    }

    if additional_params:
        for key in ("chunk_token_num", "delimiter", "overlapped_percent", "hard_limit_token_num"):
            if key in additional_params:
                result[key] = additional_params[key]

    return result
