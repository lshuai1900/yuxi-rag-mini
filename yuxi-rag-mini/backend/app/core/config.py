from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # App
    APP_NAME: str = "yuxi-rag-mini"
    DEBUG: bool = False

    # Database: sqlite or postgresql
    DB_TYPE: Literal["sqlite", "postgresql"] = "sqlite"
    SQLITE_PATH: str = "data/yuxi_rag.db"
    DATABASE_URL: str = ""

    # Milvus - use local file for Milvus Lite, or http://host:19530 for standalone
    MILVUS_URI: str = "data/milvus.db"
    MILVUS_TOKEN: str = ""
    MILVUS_DB: str = "yuxi_rag"

    # Storage: local or minio
    STORAGE_TYPE: Literal["local", "minio"] = "local"
    LOCAL_STORAGE_PATH: str = "data/files"
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "yuxi-rag"
    MINIO_SECURE: bool = False

    # Embedding
    EMBEDDING_PROVIDER: Literal["openai_compatible", "ollama", "huggingface", "fake"] = "fake"
    EMBEDDING_MODEL: str = "bge-m3"
    EMBEDDING_BASE_URL: str = "http://localhost:11434/v1"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_DIMENSION: int = 128
    EMBEDDING_BATCH_SIZE: int = 40
    EMBEDDING_TIMEOUT: int = 60
    EMBEDDING_CHUNK_SIZE: int = 512
    EMBEDDING_CHUNK_OVERLAP: int = 50

    # Rerank
    RERANK_PROVIDER: str = "dummy"

    # Milvus Query Offload
    MILVUS_QUERY_OFFLOAD_SEMAPHORE: int = 8

    # Retrieval defaults
    DEFAULT_SEARCH_MODE: str = "hybrid"
    DEFAULT_VECTOR_WEIGHT: float = 0.7
    DEFAULT_BM25_WEIGHT: float = 0.3
    DEFAULT_BM25_TOP_K: int = 20
    DEFAULT_RECALL_TOP_K: int = 20
    DEFAULT_BM25_DROP_RATIO_SEARCH: float = 0.2

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def log_config(self) -> str:
        """Return a safe summary of the current config for startup logging."""
        lines = [
            f"  APP_NAME={self.APP_NAME}",
            f"  DB_TYPE={self.DB_TYPE}",
            f"  MILVUS_URI={self.MILVUS_URI}",
            f"  STORAGE_TYPE={self.STORAGE_TYPE}",
            f"  EMBEDDING_PROVIDER={self.EMBEDDING_PROVIDER}",
            f"  EMBEDDING_MODEL={self.EMBEDDING_MODEL}",
            f"  EMBEDDING_BASE_URL={self.EMBEDDING_BASE_URL}",
            f"  EMBEDDING_DIMENSION={self.EMBEDDING_DIMENSION}",
            f"  EMBEDDING_BATCH_SIZE={self.EMBEDDING_BATCH_SIZE}",
            f"  EMBEDDING_TIMEOUT={self.EMBEDDING_TIMEOUT}",
            f"  EMBEDDING_CHUNK_SIZE={self.EMBEDDING_CHUNK_SIZE}",
            f"  EMBEDDING_CHUNK_OVERLAP={self.EMBEDDING_CHUNK_OVERLAP}",
            f"  RERANK_PROVIDER={self.RERANK_PROVIDER}",
            f"  MILVUS_QUERY_OFFLOAD_SEMAPHORE={self.MILVUS_QUERY_OFFLOAD_SEMAPHORE}",
            f"  DEFAULT_SEARCH_MODE={self.DEFAULT_SEARCH_MODE}",
            f"  DEFAULT_VECTOR_WEIGHT={self.DEFAULT_VECTOR_WEIGHT}",
            f"  DEFAULT_BM25_WEIGHT={self.DEFAULT_BM25_WEIGHT}",
            f"  DEFAULT_BM25_TOP_K={self.DEFAULT_BM25_TOP_K}",
            f"  DEFAULT_RECALL_TOP_K={self.DEFAULT_RECALL_TOP_K}",
            f"  DEFAULT_BM25_DROP_RATIO_SEARCH={self.DEFAULT_BM25_DROP_RATIO_SEARCH}",
        ]
        # Do NOT log EMBEDDING_API_KEY
        if self.EMBEDDING_PROVIDER == "fake":
            lines.append("  WARNING: FakeEmbeddingProvider is active. Only for testing, not for real RAG!")
        return "\n".join(lines)


settings = Settings()
