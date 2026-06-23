import asyncio
import os
import traceback
from dataclasses import dataclass, field, fields, MISSING
from functools import partial
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    WeightedRanker,
    connections,
    db,
    utility,
)

from app.rag.base import FileStatus, KnowledgeBase
from app.rag.chunking.text_chunker import chunk_text, count_tokens
from app.rag.chunking.markdown_chunker import chunk_markdown
from app.rag.parser.factory import parse_file
from app.rag.providers.embedding import create_embedding_provider
from app.rag.repositories.chunk_repository import KnowledgeChunkRepository
from app.rag.storage import get_file_storage
from app.core.config import settings
from app.core.logging import logger

CONTENT_SPARSE_FIELD = "content_sparse"
CONTENT_ANALYZER_PARAMS = {"type": "chinese"}
VECTOR_METRIC_TYPE = "COSINE"
MILVUS_CHUNK_EMBED_BATCH_SIZE = 200


@dataclass(kw_only=True)
class MilvusRetrievalConfig:
    search_mode: str = field(
        default="vector",
        metadata={"label": "检索模式", "type": "select",
                  "options": [
                      {"value": "vector", "label": "向量检索"},
                      {"value": "keyword", "label": "BM25 全文检索"},
                      {"value": "hybrid", "label": "混合检索"},
                  ]},
    )
    top_k: int = field(default=10, metadata={"label": "返回数量", "type": "number", "min": 1, "max": 100})
    similarity_threshold: float = field(default=0.0, metadata={"label": "相似度阈值", "type": "number", "min": 0.0, "max": 1.0})
    bm25_top_k: int = field(default=50, metadata={"label": "BM25 召回数量", "type": "number", "min": 1, "max": 200})
    vector_weight: float = field(default=0.7, metadata={"label": "向量权重", "type": "number", "min": 0.0, "max": 1.0})
    bm25_weight: float = field(default=0.3, metadata={"label": "BM25 权重", "type": "number", "min": 0.0, "max": 1.0})


class MilvusKB(KnowledgeBase):
    kb_type = "milvus"
    name = "Milvus"
    description = "Milvus-based vector knowledge base"

    def __init__(self, work_dir: str, **kwargs):
        super().__init__(work_dir)
        self.milvus_uri = kwargs.get("milvus_uri", settings.MILVUS_URI)
        self.milvus_token = kwargs.get("milvus_token", settings.MILVUS_TOKEN)
        self.milvus_db = kwargs.get("milvus_db", settings.MILVUS_DB)
        self.connection_alias = f"milvus_{id(self)}"
        self.collections: dict[str, Any] = {}
        self._metadata_lock = asyncio.Lock()
        self._connected = False
        self._try_connect()

    def _try_connect(self):
        """Try to connect to Milvus, but don't fail if unavailable."""
        try:
            connections.connect(alias=self.connection_alias, uri=self.milvus_uri, token=self.milvus_token)
            try:
                if self.milvus_db not in db.list_database():
                    db.create_database(self.milvus_db)
                db.using_database(self.milvus_db)
            except Exception as e:
                logger.warning(f"Database operation failed, using default: {e}")
            self._connected = True
            logger.info(f"Connected to Milvus at {self.milvus_uri}")
        except Exception as e:
            logger.warning(f"Milvus not available at {self.milvus_uri}: {e}. KB metadata will work, but Milvus operations will fail.")
            self._connected = False

    def _get_embedding_provider(self):
        return create_embedding_provider(
            provider=settings.EMBEDDING_PROVIDER,
            model=settings.EMBEDDING_MODEL,
            base_url=settings.EMBEDDING_BASE_URL,
            api_key=settings.EMBEDDING_API_KEY,
            dimension=settings.EMBEDDING_DIMENSION,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )

    async def _create_kb_instance(self, kb_id: str, kb_config: dict) -> Any:
        if kb_id not in self.databases_meta:
            raise ValueError(f"Database {kb_id} not found")
        embedding_provider = self._get_embedding_provider()
        collection_name = kb_id
        try:
            if utility.has_collection(collection_name, using=self.connection_alias):
                collection = Collection(name=collection_name, using=self.connection_alias)
                if self._collection_supports_bm25(collection):
                    return collection
                utility.drop_collection(collection_name, using=self.connection_alias)
            return self._create_new_collection(collection_name, embedding_provider.dimension, kb_id)
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

    def _create_new_collection(self, collection_name: str, embedding_dim: int, kb_id: str) -> Collection:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535,
                        enable_analyzer=True, analyzer_params=CONTENT_ANALYZER_PARAMS),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name=CONTENT_SPARSE_FIELD, dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        bm25_function = Function(
            name="content_bm25",
            input_field_names=["content"],
            output_field_names=[CONTENT_SPARSE_FIELD],
            function_type=FunctionType.BM25,
        )
        schema = CollectionSchema(fields=fields, functions=[bm25_function],
                                  description=f"KB collection for {kb_id}")
        collection = Collection(name=collection_name, schema=schema, using=self.connection_alias)
        index_params = {"metric_type": VECTOR_METRIC_TYPE, "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
        collection.create_index("embedding", index_params)
        sparse_index_params = {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
        }
        collection.create_index(CONTENT_SPARSE_FIELD, sparse_index_params)
        logger.info(f"Created Milvus collection: {collection_name}")
        return collection

    def _collection_supports_bm25(self, collection: Collection) -> bool:
        fields = {f.name: f for f in collection.schema.fields}
        content_field = fields.get("content")
        sparse_field = fields.get(CONTENT_SPARSE_FIELD)
        if not content_field or content_field.dtype != DataType.VARCHAR:
            return False
        if content_field.params.get("enable_analyzer") is not True:
            return False
        if not sparse_field or sparse_field.dtype != DataType.SPARSE_FLOAT_VECTOR:
            return False
        for func in collection.schema.functions:
            if (func.type == FunctionType.BM25
                    and func.input_field_names == ["content"]
                    and func.output_field_names == [CONTENT_SPARSE_FIELD]):
                return True
        return False

    async def _initialize_kb_instance(self, instance: Any) -> None:
        try:
            instance.load()
        except Exception as e:
            logger.warning(f"Failed to load collection: {e}")

    async def _get_milvus_collection(self, kb_id: str):
        if kb_id in self.collections:
            return self.collections[kb_id]
        if kb_id not in self.databases_meta:
            return None
        try:
            collection = await self._create_kb_instance(kb_id, {})
            await self._initialize_kb_instance(collection)
            self.collections[kb_id] = collection
            return collection
        except Exception as e:
            logger.error(f"Failed to get Milvus collection for {kb_id}: {e}")
            if not self._connected:
                raise RuntimeError(f"Milvus is not connected. Please check your Milvus configuration (URI: {self.milvus_uri}). Original error: {e}")
            return None

    def _split_text_into_chunks(self, text: str, file_id: str, filename: str) -> list[dict]:
        ext = os.path.splitext(filename)[1].lower() if filename else ""
        if ext in [".md", ".markdown"]:
            text_chunks = chunk_markdown(text)
        else:
            text_chunks = chunk_text(text)

        records = []
        search_from = 0
        for idx, chunk_content in enumerate(text_chunks):
            chunk_text_stripped = chunk_content.strip()
            if not chunk_text_stripped:
                continue
            start_char_pos = text.find(chunk_text_stripped, search_from)
            if start_char_pos >= 0:
                end_char_pos = start_char_pos + len(chunk_text_stripped)
                search_from = end_char_pos
            else:
                start_char_pos = None
                end_char_pos = None
            records.append({
                "id": f"{file_id}_chunk_{idx}",
                "content": chunk_text_stripped,
                "file_id": file_id,
                "filename": filename,
                "chunk_index": idx,
                "chunk_id": f"{file_id}_chunk_{idx}",
                "start_char_pos": start_char_pos,
                "end_char_pos": end_char_pos,
            })
        return records

    async def add_file_record(self, kb_id: str, item: str, params: dict | None = None, operator_id: str | None = None) -> dict:
        import hashlib
        import time
        import secrets
        import string

        params = params or {}
        file_id = "file_" + "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(10))
        filename = params.get("filename", os.path.basename(item))
        file_type = os.path.splitext(filename)[1].lower().lstrip(".")

        metadata = {
            "file_id": file_id,
            "kb_id": kb_id,
            "filename": filename,
            "file_type": file_type,
            "path": item,
            "status": FileStatus.UPLOADED,
            "created_at": _utc_now(),
            "size": params.get("size", 0),
            "content_hash": params.get("content_hash"),
        }
        self.files_meta[file_id] = metadata
        await self._persist_file(file_id)
        await self.refresh_database_stats(kb_id)
        return metadata

    async def parse_file(self, kb_id: str, file_id: str, operator_id: str | None = None) -> dict:
        if file_id not in self.files_meta:
            raise ValueError(f"File {file_id} not found")
        file_meta = self.files_meta[file_id]

        self.files_meta[file_id]["status"] = FileStatus.PARSING
        await self._persist_file(file_id)

        try:
            file_path = file_meta.get("path")
            storage = get_file_storage()

            # Resolve the actual file path for parsing
            if file_path.startswith("local://"):
                actual_path = storage.get_local_path(file_path.replace("local://", ""))
            elif file_path.startswith("minio://"):
                data = await storage.download_file(file_path.split("/", 3)[-1])
                import tempfile
                suffix = os.path.splitext(file_meta.get("filename", ""))[1]
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(data)
                    actual_path = tmp.name
            else:
                # Assume it's a local storage object name
                actual_path = storage.get_local_path(file_path)

            markdown_content = await parse_file(actual_path)

            # Save parsed markdown
            parsed_object_name = f"{kb_id}/parsed/{file_id}.md"
            await storage.upload_file(parsed_object_name, markdown_content.encode("utf-8"), "text/markdown")

            self.files_meta[file_id]["status"] = FileStatus.PARSED
            self.files_meta[file_id]["markdown_file"] = parsed_object_name
            await self._persist_file(file_id)
            return self.files_meta[file_id]
        except Exception as e:
            logger.error(f"Failed to parse file {file_id}: {e}")
            self.files_meta[file_id]["status"] = FileStatus.ERROR_PARSING
            self.files_meta[file_id]["error"] = str(e)
            await self._persist_file(file_id)
            raise

    async def index_file(self, kb_id: str, file_id: str, operator_id: str | None = None) -> dict:
        if kb_id not in self.databases_meta:
            raise ValueError(f"Database {kb_id} not found")

        collection = await self._get_milvus_collection(kb_id)
        if not collection:
            raise ValueError(f"Failed to get Milvus collection for {kb_id}")

        if file_id not in self.files_meta:
            raise ValueError(f"File {file_id} not found")
        file_meta = self.files_meta[file_id]

        if not file_meta.get("markdown_file"):
            raise ValueError("File has not been parsed yet")

        self.files_meta[file_id]["status"] = FileStatus.INDEXING
        await self._persist_file(file_id)

        try:
            storage = get_file_storage()
            markdown_path = file_meta["markdown_file"]
            if markdown_path.startswith("local://"):
                actual_path = storage.get_local_path(markdown_path.replace("local://", ""))
                with open(actual_path, encoding="utf-8") as f:
                    markdown_content = f.read()
            else:
                data = await storage.download_file(markdown_path)
                markdown_content = data.decode("utf-8")

            filename = file_meta.get("filename", "")
            chunks = self._split_text_into_chunks(markdown_content, file_id, filename)
            logger.info(f"Split {filename} into {len(chunks)} chunks")

            # Delete existing chunks
            await self._delete_file_chunks(kb_id, file_id, collection)

            if chunks:
                # Embed and store
                embedding_provider = self._get_embedding_provider()
                texts = [c["content"] for c in chunks]
                embeddings = await embedding_provider.abatch_encode(texts)

                # Insert to Milvus
                entities = [
                    [c["id"] for c in chunks],
                    [c["content"] for c in chunks],
                    [c["chunk_id"] for c in chunks],
                    [c["file_id"] for c in chunks],
                    [c["chunk_index"] for c in chunks],
                    embeddings,
                ]
                await asyncio.to_thread(collection.insert, entities)

                # Insert to DB
                chunk_repo = KnowledgeChunkRepository()
                chunk_records = [
                    {
                        "chunk_id": c["chunk_id"],
                        "file_id": c["file_id"],
                        "kb_id": kb_id,
                        "chunk_index": c["chunk_index"],
                        "content": c["content"],
                        "start_char_pos": c.get("start_char_pos"),
                        "end_char_pos": c.get("end_char_pos"),
                    }
                    for c in chunks
                ]
                await chunk_repo.batch_upsert(chunk_records)

            chunk_stats = {
                "chunk_count": len(chunks),
                "token_count": sum(count_tokens(c["content"]) for c in chunks),
            }
            self.files_meta[file_id]["status"] = FileStatus.INDEXED
            self.files_meta[file_id].update(chunk_stats)
            await self._persist_file(file_id)
            await self.refresh_database_stats(kb_id)
            return self.files_meta[file_id]

        except Exception as e:
            logger.error(f"Indexing failed for {file_id}: {e}")
            self.files_meta[file_id]["status"] = FileStatus.ERROR_INDEXING
            self.files_meta[file_id]["error"] = str(e)
            await self._persist_file(file_id)
            raise

    async def _delete_file_chunks(self, kb_id: str, file_id: str, collection: Collection) -> None:
        chunk_repo = KnowledgeChunkRepository()
        await chunk_repo.delete_by_file_id(file_id)
        try:
            expr = f'file_id == "{file_id}"'
            results = await asyncio.to_thread(collection.query, expr=expr, output_fields=["id"], limit=1)
            if results:
                await asyncio.to_thread(collection.delete, expr)
        except Exception as e:
            logger.warning(f"Error deleting chunks from Milvus: {e}")

    async def aquery(self, query_text: str, kb_id: str, **kwargs) -> list[dict]:
        collection = await self._get_milvus_collection(kb_id)
        if not collection:
            raise ValueError(f"Database {kb_id} not found")

        search_mode = str(kwargs.get("search_mode", "vector")).lower()
        if search_mode not in {"vector", "keyword", "hybrid"}:
            search_mode = "vector"
        top_k = int(kwargs.get("top_k", 10))
        similarity_threshold = float(kwargs.get("similarity_threshold", 0.0))

        output_fields = ["content", "chunk_id", "file_id", "chunk_index"]
        retrieved_chunks = []

        try:
            if search_mode == "vector":
                embedding_provider = self._get_embedding_provider()
                query_embedding = await embedding_provider.abatch_encode([query_text])
                search_params = {"metric_type": VECTOR_METRIC_TYPE, "params": {"nprobe": 10}}
                results = await asyncio.to_thread(
                    collection.search,
                    data=query_embedding,
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    output_fields=output_fields,
                )
                if results and len(results) > 0:
                    for hit in results[0]:
                        similarity = hit.distance
                        if similarity < similarity_threshold:
                            continue
                        entity = hit.entity
                        retrieved_chunks.append({
                            "content": entity.get("content", ""),
                            "metadata": {
                                "source": self._get_filename(entity.get("file_id")),
                                "chunk_id": entity.get("chunk_id"),
                                "file_id": entity.get("file_id"),
                                "chunk_index": entity.get("chunk_index"),
                            },
                            "score": float(similarity),
                        })

            elif search_mode == "keyword":
                bm25_top_k = int(kwargs.get("bm25_top_k", top_k))
                bm25_search_params = {"metric_type": "BM25", "params": {"drop_ratio_search": 0.0}}
                results = await asyncio.to_thread(
                    collection.search,
                    data=[query_text],
                    anns_field=CONTENT_SPARSE_FIELD,
                    param=bm25_search_params,
                    limit=bm25_top_k,
                    output_fields=output_fields,
                )
                if results and len(results) > 0:
                    for hit in results[0]:
                        entity = hit.entity
                        retrieved_chunks.append({
                            "content": entity.get("content", ""),
                            "metadata": {
                                "source": self._get_filename(entity.get("file_id")),
                                "chunk_id": entity.get("chunk_id"),
                                "file_id": entity.get("file_id"),
                                "chunk_index": entity.get("chunk_index"),
                            },
                            "score": float(hit.distance),
                        })

            elif search_mode == "hybrid":
                embedding_provider = self._get_embedding_provider()
                query_embedding = await embedding_provider.abatch_encode([query_text])
                bm25_top_k = int(kwargs.get("bm25_top_k", top_k))
                vector_weight = float(kwargs.get("vector_weight", 0.7))
                bm25_weight = float(kwargs.get("bm25_weight", 0.3))

                vector_request = AnnSearchRequest(
                    data=query_embedding,
                    anns_field="embedding",
                    param={"metric_type": VECTOR_METRIC_TYPE, "params": {"nprobe": 10}},
                    limit=top_k,
                )
                bm25_request = AnnSearchRequest(
                    data=[query_text],
                    anns_field=CONTENT_SPARSE_FIELD,
                    param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.0}},
                    limit=bm25_top_k,
                )
                results = await asyncio.to_thread(
                    collection.hybrid_search,
                    reqs=[vector_request, bm25_request],
                    rerank=WeightedRanker(vector_weight, bm25_weight),
                    limit=top_k,
                    output_fields=output_fields,
                )
                if results and len(results) > 0:
                    for hit in results[0]:
                        score = float(hit.distance or 0.0)
                        if score < similarity_threshold:
                            continue
                        entity = hit.entity
                        retrieved_chunks.append({
                            "content": entity.get("content", ""),
                            "metadata": {
                                "source": self._get_filename(entity.get("file_id")),
                                "chunk_id": entity.get("chunk_id"),
                                "file_id": entity.get("file_id"),
                                "chunk_index": entity.get("chunk_index"),
                            },
                            "score": score,
                        })

            return retrieved_chunks[:top_k]

        except Exception as e:
            logger.error(f"Milvus query error: {e}")
            return []

    def _get_filename(self, file_id: str | None) -> str:
        if not file_id:
            return "unknown"
        return self.files_meta.get(file_id, {}).get("filename") or "unknown"

    async def delete_file(self, kb_id: str, file_id: str) -> None:
        collection = await self._get_milvus_collection(kb_id)
        if collection:
            await self._delete_file_chunks(kb_id, file_id, collection)
        if file_id in self.files_meta:
            del self.files_meta[file_id]
            from app.rag.repositories.file_repository import KnowledgeFileRepository
            await KnowledgeFileRepository().delete(file_id)
        await self.refresh_database_stats(kb_id)

    async def get_file_basic_info(self, kb_id: str, file_id: str) -> dict:
        if file_id not in self.files_meta:
            raise ValueError(f"File {file_id} not found")
        return {"meta": self.files_meta[file_id]}

    async def get_file_content(self, kb_id: str, file_id: str) -> dict:
        if file_id not in self.files_meta:
            raise ValueError(f"File {file_id} not found")
        chunk_repo = KnowledgeChunkRepository()
        chunks = await chunk_repo.list_by_file_id(file_id)
        return {
            "lines": [
                {
                    "id": c.chunk_id,
                    "content": c.content,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ]
        }

    async def get_file_info(self, kb_id: str, file_id: str) -> dict:
        basic = await self.get_file_basic_info(kb_id, file_id)
        content = await self.get_file_content(kb_id, file_id)
        return {**basic, **content}

    async def delete_database(self, kb_id: str) -> dict:
        try:
            if utility.has_collection(kb_id, using=self.connection_alias):
                utility.drop_collection(kb_id, using=self.connection_alias)
        except Exception as e:
            logger.error(f"Failed to drop collection {kb_id}: {e}")
        return await super().delete_database(kb_id)

    def __del__(self):
        try:
            if hasattr(self, "connection_alias"):
                connections.disconnect(self.connection_alias)
        except Exception:
            pass


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
