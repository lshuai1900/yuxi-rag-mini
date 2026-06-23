from dataclasses import dataclass
from typing import Any, Literal
from pydantic import BaseModel, Field


class ScoreDetail(BaseModel):
    """Detailed score breakdown for hybrid search results."""
    vector_score: float = 0.0
    keyword_score: float = 0.0
    final_score: float = 0.0
    source: str = "vector"  # vector, keyword, hybrid


class SearchResultSchema(BaseModel):
    chunk_id: str = Field(description="chunk ID")
    file_id: str = Field(default="", description="file ID")
    filename: str = Field(default="", description="source filename")
    content: str = Field(description="chunk content")
    chunk_index: int = Field(default=0)
    score: float = Field(default=0.0, description="relevance score")
    score_detail: ScoreDetail | None = Field(default=None, description="score breakdown for hybrid search")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankInfo(BaseModel):
    """Rerank information in query response."""
    reranked: bool = False
    reranker: str = ""


class QueryResponse(BaseModel):
    query: str = Field(description="original query text")
    search_mode: str = Field(description="search mode used")
    results: list[SearchResultSchema] = Field(default_factory=list)
    rerank: RerankInfo | None = None


class QueryRequest(BaseModel):
    query: str = Field(description="query text")
    search_mode: Literal["vector", "keyword", "hybrid"] = "vector"
    top_k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    enable_rerank: bool = Field(default=False, description="enable rerank (DummyReranker only)")
    enable_graphrag: bool = Field(default=False, description="enable GraphRAG (not implemented)")
    retrieval_config: dict[str, Any] | None = None


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
    failed_reason: str = ""
    created_at: str | None = None


class IndexResponse(BaseModel):
    file_id: str
    filename: str
    chunk_count: int
    status: str
    failed_reason: str = ""


class ErrorDetail(BaseModel):
    """Unified error response format."""
    code: str = "INTERNAL_ERROR"
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass
class MilvusRetrievalConfig:
    search_mode: str = "hybrid"
    final_top_k: int = 10
    similarity_threshold: float = 0.0
    bm25_top_k: int = 20
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    bm25_drop_ratio_search: float = 0.2
    include_distances: bool = True
    use_reranker: bool = False
    reranker_model: str = ""
    recall_top_k: int = 20
    use_graph_retrieval: bool = False
    graph_entity_top_k: int = 5
    graph_triple_top_k: int = 5
    graph_max_nodes: int = 100
    graph_top_k: int = 10
    graph_weight: float = 0.1
    ppr_damping: float = 0.85
