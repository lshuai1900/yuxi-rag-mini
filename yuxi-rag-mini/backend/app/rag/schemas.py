from typing import Any, Literal
from pydantic import BaseModel, Field


class SearchResultSchema(BaseModel):
    id: str = Field(description="chunk ID")
    kb_id: str = Field(description="knowledge base ID")
    file_id: str = Field(default="", description="file ID")
    content: str = Field(description="chunk content")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchOutputSchema(BaseModel):
    kb_id: str
    results: list[SearchResultSchema] = Field(default_factory=list)


class SearchInputSchema(BaseModel):
    kb_id: str
    query_text: str
    search_mode: Literal["vector", "keyword", "hybrid"] = "vector"
    top_k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    file_name: str | None = None


class KBCreateSchema(BaseModel):
    name: str
    description: str = ""
    embedding_model_spec: str | None = None
    additional_params: dict[str, Any] | None = None


class KBInfoSchema(BaseModel):
    kb_id: str
    name: str
    description: str = ""
    kb_type: str = "milvus"
    embedding_model_spec: str | None = None
    stats: dict[str, int] = {}
    created_at: str | None = None


class FileInfoSchema(BaseModel):
    file_id: str
    kb_id: str
    filename: str
    file_type: str = ""
    status: str = "uploaded"
    size: int = 0
    chunk_count: int = 0
    token_count: int = 0
    created_at: str | None = None
