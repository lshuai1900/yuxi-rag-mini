from typing import Any, Literal
from pydantic import BaseModel, Field


class SearchResultSchema(BaseModel):
    chunk_id: str = Field(description="chunk ID")
    file_id: str = Field(default="", description="file ID")
    filename: str = Field(default="", description="source filename")
    content: str = Field(description="chunk content")
    score: float = Field(default=0.0, description="relevance score")
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str = Field(description="original query text")
    search_mode: str = Field(description="search mode used")
    results: list[SearchResultSchema] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str = Field(description="query text")
    search_mode: Literal["vector", "keyword", "hybrid"] = "vector"
    top_k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    enable_rerank: bool = Field(default=False, description="enable rerank (DummyReranker only)")
    enable_graphrag: bool = Field(default=False, description="enable GraphRAG (not implemented)")


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


class IndexResponse(BaseModel):
    file_id: str
    filename: str
    chunk_count: int
    status: str
