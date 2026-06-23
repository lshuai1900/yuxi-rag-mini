import asyncio
import json
import os
from dataclasses import asdict
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
from app.rag.chunking.ragflow_like import chunk_markdown as ragflow_chunk_markdown
from app.rag.parser.unified import parse_source_to_markdown
from app.rag.schemas import MilvusRetrievalConfig
from app.rag.providers.embedding import create_embedding_provider
from app.rag.providers.rerank import create_reranker
from app.rag.repositories.chunk_repository import KnowledgeChunkRepository
from app.rag.storage import get_file_storage
from app.core.config import settings
from app.core.logging import logger

VECTOR_METRIC_TYPE = "COSINE"

# Query offload semaphore for limiting concurrent Milvus queries
_query_semaphores: dict[str, asyncio.Semaphore] = {}


def _get_milvus_query_offload_semaphore() -> asyncio.Semaphore:
    """Get or create a semaphore for Milvus query offloading."""
    key = "default"
    if key not in _query_semaphores:
        _query_semaphores[key] = asyncio.Semaphore(settings.MILVUS_QUERY_OFFLOAD_SEMAPHORE)
    return _query_semaphores[key]


async def _run_milvus_query_io(func, /, *args, **kwargs):
    """Run a blocking Milvus query in a thread pool with semaphore-based concurrency control."""
    semaphore = _get_milvus_query_offload_semaphore()
    await semaphore.acquire()
    try:
        task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
        result = await task
        return result
    finally:
        semaphore.release()


class MilvusKB(KnowledgeBase):
    kb_type = "milvus"
    name = "Milvus"
    description = "Milvus-based vector knowledge base with native BM25 and hybrid search"

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
                try:
                    collection.release()
                except Exception:
                    pass
                utility.drop_collection(collection_name, using=self.connection_alias)
            return self._create_new_collection(collection_name, embedding_provider.dimension, kb_id)
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

    def _create_new_collection(self, collection_name: str, embedding_dim: int, kb_id: str) -> Collection:
        """Create a new Milvus collection with BM25 sparse vector support.

        Schema aligns with Yuxi's milvus.py implementation:
        - content field with Chinese analyzer for BM25
        - content_sparse SPARSE_FLOAT_VECTOR via FunctionType.BM25
        - embedding FLOAT_VECTOR for dense vectors
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(
                name="content", dtype=DataType.VARCHAR, max_length=65535,
                enable_analyzer=True, analyzer_params={"type": "chinese"},
            ),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="content_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]

        # BM25 function: generates sparse vector from text content
        bm25_function = Function(
            name="content_bm25",
            input_field_names=["content"],
            output_field_names=["content_sparse"],
            function_type=FunctionType.BM25,
        )

        schema = CollectionSchema(
            fields=fields,
            functions=[bm25_function],
            description=f"KB collection for {kb_id} (with BM25)",
            enable_dynamic_field=False,
        )
        collection = Collection(name=collection_name, schema=schema, using=self.connection_alias)

        # Dense vector index (IVF_FLAT or HNSW depending on scale; use IVF_FLAT for compatibility)
        dense_index_params = {
            "metric_type": VECTOR_METRIC_TYPE,
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index("embedding", dense_index_params, index_name="embedding_idx")

        # Sparse vector index for BM25 (SPARSE_INVERTED_INDEX with DAAT_MAXSCORE)
        sparse_index_params = {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
        }
        collection.create_index("content_sparse", sparse_index_params, index_name="sparse_idx")

        logger.info(f"Created Milvus collection with BM25 support: {collection_name} (dim={embedding_dim})")
        return collection

    @staticmethod
    def _validate_collection_schema(collection: Collection, expected_dim: int) -> bool:
        """Validate that collection schema supports BM25 and matches embedding dimension.

        Returns True if valid, False otherwise.
        """
        try:
            schema = collection.schema
            has_bm25 = any(
                func.function_type == FunctionType.BM25
                for func in (schema.functions or [])
            )
            if not has_bm25:
                logger.warning("Collection does not have BM25 function")

            for field in schema.fields:
                if field.name == "embedding":
                    actual_dim = field.params.get("dim")
                    if actual_dim != expected_dim:
                        logger.warning(f"Embedding dim mismatch: collection={actual_dim}, config={expected_dim}")
                        return False
                    break

            return True
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False

    async def _initialize_kb_instance(self, instance: Any) -> None:
        try:
            instance.load()
        except Exception as e:
            logger.warning(f"Failed to load collection: {e}")

    async def _get_or_create_collection(self, kb_id: str):
        """Get existing collection or create new one for kb_id.

        Each KB gets its own collection (not shared).
        """
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

    async def add_file_record(self, kb_id: str, item: str, params: dict | None = None,
                              operator_id: str | None = None) -> dict:
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
        """Parse file to markdown and save markdown_file.

        Aligns with Yuxi's KnowledgeBase.parse_file():
        1. Status UPLOADED -> PARSING
        2. Parse source to markdown using unified parser
        3. Save markdown_file to storage
        4. Status PARSING -> PARSED (or ERROR_PARSING on failure)
        """
        if file_id not in self.files_meta:
            raise ValueError(f"File {file_id} not found")
        file_meta = self.files_meta[file_id]

        self.files_meta[file_id]["status"] = FileStatus.PARSING
        await self._persist_file(file_id)

        try:
            file_path = file_meta.get("path")
            storage = get_file_storage()

            # Resolve actual path
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
                actual_path = storage.get_local_path(file_path)

            filename = file_meta.get("filename", "")

            # Use unified parser: converts file to markdown
            parse_result = await parse_source_to_markdown(actual_path, params={"filename": filename})

            # Check for empty parse result
            if not parse_result.markdown or not parse_result.markdown.strip():
                raise ValueError(f"No extractable content found in {filename}. The file may be empty or contain only non-text content.")

            # Save parsed markdown content
            parsed_object_name = f"{kb_id}/parsed/{file_id}.md"
            await storage.upload_file(parsed_object_name, parse_result.markdown.encode("utf-8"), "text/markdown")

            self.files_meta[file_id]["status"] = FileStatus.PARSED
            self.files_meta[file_id]["markdown_file"] = parsed_object_name
            self.files_meta[file_id]["parse_metadata"] = {
                "file_ext": parse_result.file_ext,
                "artifacts": parse_result.artifacts,
            }
            await self._persist_file(file_id)
            return self.files_meta[file_id]

        except Exception as e:
            logger.error(f"Failed to parse file {file_id}: {e}")
            self.files_meta[file_id]["status"] = FileStatus.ERROR_PARSING
            self.files_meta[file_id]["failed_reason"] = str(e)
            self.files_meta[file_id]["failed_stage"] = "parsing"
            self.files_meta[file_id]["failed_at"] = _utc_now()
            await self._persist_file(file_id)
            raise

    async def index_file(self, kb_id: str, file_id: str, operator_id: str | None = None) -> dict:
        """Index a parsed file: read markdown -> chunk -> embed -> insert Milvus + PG.

        Aligns with Yuxi's KnowledgeBase.index_file():
        1. Status PARSED -> INDEXING
        2. Read markdown_file from storage
        3. Chunk markdown using ragflow_like dispatcher
        4. Embed chunks
        5. Double-write: Milvus insert + PostgreSQL upsert (with rollback on failure)
        6. Status INDEXING -> INDEXED (or ERROR_INDEXING on failure)
        """
        if kb_id not in self.databases_meta:
            raise ValueError(f"Database {kb_id} not found")

        if not self._connected:
            raise RuntimeError("Milvus is not connected. Cannot index files.")

        collection = await self._get_or_create_collection(kb_id)
        if not collection:
            raise RuntimeError(f"Failed to get Milvus collection for {kb_id}")

        if file_id not in self.files_meta:
            raise ValueError(f"File {file_id} not found")
        file_meta = self.files_meta[file_id]

        if not file_meta.get("markdown_file"):
            raise ValueError("File has not been parsed yet. Call parse first.")

        # Stage: indexing (encompasses chunking + embedding + writing)
        self.files_meta[file_id]["status"] = FileStatus.INDEXING
        await self._persist_file(file_id)

        try:
            storage = get_file_storage()
            markdown_path = file_meta["markdown_file"]
            if markdown_path.startswith("local://"):
                actual_path = storage.get_local_path(markdown_path.replace("local://", ""))
                with open(actual_path, encoding="utf-8") as f:
                    text_content = f.read()
            else:
                data = await storage.download_file(markdown_path)
                text_content = data.decode("utf-8")

            filename = file_meta.get("filename", "")
            processing_params = file_meta.get("processing_params", {})

            # Chunk using ragflow_like dispatcher (returns list of dicts with position info)
            chunks = ragflow_chunk_markdown(
                text_content,
                file_id=file_id,
                filename=filename,
                processing_params=processing_params,
            )
            logger.info(f"Split {filename} into {len(chunks)} chunks (ragflow_like)")

            if not chunks:
                raise ValueError(f"No chunks generated from {filename}. The file content may be too short or empty.")

            # Delete existing chunks for this file (both Milvus and PG)
            await self._delete_file_chunks_from_milvus(collection, file_id)
            chunk_repo = KnowledgeChunkRepository()
            await chunk_repo.delete_by_file_id(file_id)

            # Embed chunks
            embedding_provider = self._get_embedding_provider()
            texts = [c["content"] for c in chunks]
            embeddings = await embedding_provider.embed_documents(texts)
            logger.info(f"Embedded {len(texts)} chunks")

            # Prepare entities for Milvus insert
            ids = [c["id"] for c in chunks]
            contents = [c["content"] for c in chunks]
            chunk_ids = [c["chunk_id"] for c in chunks]
            file_ids = [c["file_id"] for c in chunks]
            chunk_indices = [c["chunk_index"] for c in chunks]

            entities = [ids, contents, chunk_ids, file_ids, chunk_indices, embeddings]

            # Double-write: Milvus insert + PostgreSQL upsert with rollback on failure
            pg_records = [
                {
                    "chunk_id": c["chunk_id"],
                    "file_id": c["file_id"],
                    "kb_id": kb_id,
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "start_char_pos": c.get("start_char_pos"),
                    "end_char_pos": c.get("end_char_pos"),
                    "start_token_pos": c.get("start_token_pos"),
                    "end_token_pos": c.get("end_token_pos"),
                }
                for c in chunks
            ]

            pg_task = asyncio.create_task(chunk_repo.batch_upsert(pg_records))
            milvus_task = asyncio.create_task(asyncio.to_thread(collection.insert, entities))

            results = await asyncio.gather(pg_task, milvus_task, return_exceptions=True)
            errors = [r for r in results if isinstance(r, Exception)]

            if errors:
                # Rollback: delete from both stores
                logger.error(f"Double-write failed, rolling back: {errors[0]}")
                try:
                    await chunk_repo.delete_by_file_id(file_id)
                except Exception as rb_err:
                    logger.warning(f"PG rollback failed: {rb_err}")
                try:
                    await self._delete_file_chunks_from_milvus(collection, file_id)
                except Exception as rb_err:
                    logger.warning(f"Milvus rollback failed: {rb_err}")
                raise errors[0]

            await asyncio.to_thread(collection.flush)
            logger.info(f"Inserted {len(chunks)} chunks into Milvus (double-write successful)")

            chunk_stats = {
                "chunk_count": len(chunks),
                "token_count": sum(_count_tokens(c["content"]) for c in chunks),
            }
            self.files_meta[file_id]["status"] = FileStatus.INDEXED
            self.files_meta[file_id].update(chunk_stats)
            # Clear any previous failure info on success
            self.files_meta[file_id].pop("failed_reason", None)
            self.files_meta[file_id].pop("failed_stage", None)
            self.files_meta[file_id].pop("failed_at", None)
            await self._persist_file(file_id)
            await self.refresh_database_stats(kb_id)

            return {
                "file_id": file_id,
                "filename": filename,
                "chunk_count": len(chunks),
                "status": FileStatus.INDEXED,
            }

        except Exception as e:
            logger.error(f"Indexing failed for {file_id}: {e}")
            self.files_meta[file_id]["status"] = FileStatus.ERROR_INDEXING
            self.files_meta[file_id]["failed_reason"] = str(e)
            self.files_meta[file_id]["failed_stage"] = "indexing"
            self.files_meta[file_id]["failed_at"] = _utc_now()
            await self._persist_file(file_id)
            raise

    async def _delete_file_chunks_from_milvus(self, collection: Collection, file_id: str) -> None:
        """Delete all chunks for a file from Milvus collection."""
        try:
            expr = f'file_id == "{file_id}"'
            await _run_milvus_query_io(collection.delete, expr)
        except Exception as e:
            logger.warning(f"Error deleting chunks from Milvus: {e}")

    async def aquery(self, query_text: str, kb_id: str, **kwargs) -> list[dict]:
        """Query knowledge base with retrieval config support.

        Supports Milvus native hybrid search with WeightedRanker.
        Falls back gracefully when BM25 is unavailable.
        """
        # Build retrieval config from kwargs
        rc_dict = kwargs.pop("retrieval_config", None)
        if isinstance(rc_dict, dict):
            config = MilvusRetrievalConfig(**{k: v for k, v in rc_dict.items() if hasattr(MilvusRetrievalConfig, k)})
        else:
            config = MilvusRetrievalConfig(
                search_mode=str(kwargs.get("search_mode", settings.DEFAULT_SEARCH_MODE)),
                final_top_k=int(kwargs.get("top_k", 10)),
                similarity_threshold=float(kwargs.get("similarity_threshold", 0.0)),
            )

        search_mode = config.search_mode.lower()
        if search_mode not in ("vector", "keyword", "hybrid"):
            search_mode = "hybrid"

        if search_mode == "hybrid":
            return await self._native_hybrid_search(query_text, kb_id, config)
        elif search_mode == "vector":
            return await self._vector_search(query_text, kb_id, config)
        elif search_mode == "keyword":
            return await self._bm25_search(query_text, kb_id, config)
        return []

    async def _vector_search(self, query_text: str, kb_id: str,
                             config: MilvusRetrievalConfig) -> list[dict]:
        """Pure vector search using dense embeddings."""
        if not self._connected:
            raise RuntimeError("Milvus is not connected.")
        collection = await self._get_or_create_collection(kb_id)
        if not collection:
            raise RuntimeError(f"Collection for {kb_id} not found")

        embedding_provider = self._get_embedding_provider()
        query_embedding = await embedding_provider.embed_query(query_text)
        output_fields = ["content", "chunk_id", "file_id", "chunk_index"]

        results = await _run_milvus_query_io(
            collection.search,
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": VECTOR_METRIC_TYPE, "params": {"nprobe": 10}},
            limit=config.final_top_k,
            output_fields=output_fields,
        )

        retrieved = []
        if results and len(results) > 0:
            for hit in results[0]:
                score = float(hit.distance) if hit.distance is not None else 0.0
                if hit.distance is None:
                    score = round(1.0 - (len(retrieved) * 0.1), 4)
                if config.similarity_threshold > 0 and score < config.similarity_threshold:
                    continue
                entity = hit.entity
                retrieved.append({
                    "chunk_id": entity.get("chunk_id", ""),
                    "file_id": entity.get("file_id", ""),
                    "filename": self._get_filename(entity.get("file_id")),
                    "content": entity.get("content", ""),
                    "score": score,
                    "chunk_index": entity.get("chunk_index", 0),
                    "score_detail": {
                        "vector_score": score,
                        "keyword_score": 0.0,
                        "final_score": score,
                        "source": "vector",
                    },
                })
        return retrieved

    async def _bm25_search(self, query_text: str, kb_id: str,
                           config: MilvusRetrievalConfig) -> list[dict]:
        """BM25 keyword search using Milvus native sparse vector."""
        if not self._connected:
            raise RuntimeError("Milvus is not connected.")
        collection = await self._get_or_create_collection(kb_id)
        if not collection:
            raise RuntimeError(f"Collection for {kb_id} not found")

        output_fields = ["content", "chunk_id", "file_id", "chunk_index"]

        results = await _run_milvus_query_io(
            collection.search,
            data=[query_text],
            anns_field="content_sparse",
            param={
                "metric_type": "BM25",
                "params": {"drop_ratio_search": config.bm25_drop_ratio_search},
            },
            limit=config.bm25_top_k,
            output_fields=output_fields,
        )

        retrieved = []
        if results and len(results) > 0:
            for hit in results[0]:
                score = float(hit.distance) if hit.distance is not None else 0.0
                if config.similarity_threshold > 0 and score < config.similarity_threshold:
                    continue
                entity = hit.entity
                retrieved.append({
                    "chunk_id": entity.get("chunk_id", ""),
                    "file_id": entity.get("file_id", ""),
                    "filename": self._get_filename(entity.get("file_id")),
                    "content": entity.get("content", ""),
                    "score": score,
                    "chunk_index": entity.get("chunk_index", 0),
                    "score_detail": {
                        "vector_score": 0.0,
                        "keyword_score": score,
                        "final_score": score,
                        "source": "keyword",
                    },
                })
        return retrieved[:config.final_top_k]

    async def _native_hybrid_search(self, query_text: str, kb_id: str,
                                    config: MilvusRetrievalConfig) -> list[dict]:
        """Milvus native hybrid search using AnnSearchRequest + WeightedRanker.

        This is the primary search mode that combines:
        1. Dense vector search (COSINE distance on embeddings)
        2. BM25 sparse vector search (on content_sparse field)
        3. WeightedRanker fusion with configurable weights
        """
        if not self._connected:
            raise RuntimeError("Milvus is not connected.")
        collection = await self._get_or_create_collection(kb_id)
        if not collection:
            raise RuntimeError(f"Collection for {kb_id} not found")

        embedding_provider = self._get_embedding_provider()
        query_embedding = await embedding_provider.embed_query(query_text)
        recall_top_k = config.recall_top_k
        output_fields = ["content", "chunk_id", "file_id", "chunk_index"]

        # Build AnnSearchRequest for dense vector search
        vector_request = AnnSearchRequest(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": VECTOR_METRIC_TYPE, "params": {"nprobe": 10}},
            limit=recall_top_k,
            expr=None,
        )

        # Build AnnSearchRequest for BM25 sparse vector search
        bm25_request = AnnSearchRequest(
            data=[query_text],
            anns_field="content_sparse",
            param={"metric_type": "BM25", "params": {"drop_ratio_search": config.bm25_drop_ratio_search}},
            limit=config.bm25_top_k,
            expr=None,
        )

        # Execute hybrid search with WeightedRanker
        try:
            results = await _run_milvus_query_io(
                collection.hybrid_search,
                reqs=[vector_request, bm25_request],
                rerank=WeightedRanker(config.vector_weight, config.bm25_weight),
                limit=config.recall_top_k,
                output_fields=output_fields,
            )
        except Exception as e:
            logger.warning(f"Native hybrid search failed ({e}), falling back to manual merge")
            # Fallback: run searches separately and merge manually
            return await self._fallback_hybrid_merge(query_text, kb_id, collection, config)

        retrieved = []
        if results and len(results) > 0:
            for hit in results[0]:
                score = float(hit.distance) if hit.distance is not None else 0.0
                if config.similarity_threshold > 0 and score < config.similarity_threshold:
                    continue
                entity = hit.entity
                retrieved.append({
                    "chunk_id": entity.get("chunk_id", ""),
                    "file_id": entity.get("file_id", ""),
                    "filename": self._get_filename(entity.get("file_id")),
                    "content": entity.get("content", ""),
                    "score": score,
                    "chunk_index": entity.get("chunk_index", 0),
                    "score_detail": {
                        "vector_score": 0.0,  # WeightedRanker produces fused scores
                        "keyword_score": 0.0,
                        "final_score": score,
                        "source": "hybrid",
                    },
                })

        # Apply final_top_k truncation
        return retrieved[:config.final_top_k]

    async def _fallback_hybrid_merge(self, query_text: str, kb_id: str,
                                     collection: Collection,
                                     config: MilvusRetrievalConfig) -> list[dict]:
        """Fallback: manually merge vector + BM25 results when native hybrid fails."""
        vector_results = []
        bm25_results = []

        try:
            vector_results = await self._vector_search(query_text, kb_id, config)
        except Exception as e:
            logger.warning(f"Fallback vector search failed: {e}")

        try:
            bm25_results = await self._bm25_search(query_text, kb_id, config)
        except Exception as e:
            logger.warning(f"Fallback BM25 search failed: {e}")

        # Merge by chunk_id, weighted sum of normalized scores
        vec_scores = [(r["chunk_id"], r["score"]) for r in vector_results]
        bm25_scores = [(r["chunk_id"], r["score"]) for r in bm25_results]

        merged: dict[str, dict] = {}

        for cid, vscore in vec_scores:
            merged[cid] = next((r for r in vector_results if r["chunk_id"] == cid), {})
            merged[cid]["score_detail"] = {
                "vector_score": vscore,
                "keyword_score": 0.0,
                "final_score": vscore * config.vector_weight,
                "source": "hybrid",
            }
            merged[cid]["score"] = vscore * config.vector_weight

        for cid, bscore in bm25_scores:
            weighted_b = bscore * config.bm25_weight
            if cid in merged:
                merged[cid]["score"] += weighted_b
                merged[cid]["score_detail"]["keyword_score"] = bscore
                merged[cid]["score_detail"]["final_score"] = merged[cid]["score"]
            else:
                rec = next((r for r in bm25_results if r["chunk_id"] == cid), {})
                merged[cid] = {**rec}
                merged[cid]["score_detail"] = {
                    "vector_score": 0.0,
                    "keyword_score": bscore,
                    "final_score": weighted_b,
                    "source": "hybrid",
                }
                merged[cid]["score"] = weighted_b

        results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        if config.similarity_threshold > 0:
            results = [r for r in results if r["score"] >= config.similarity_threshold]
        return results[:config.final_top_k]

    def _get_filename(self, file_id: str | None) -> str:
        if not file_id:
            return "unknown"
        return self.files_meta.get(file_id, {}).get("filename") or "unknown"

    async def delete_file(self, kb_id: str, file_id: str) -> None:
        collection = await self._get_or_create_collection(kb_id)
        if collection:
            await self._delete_file_chunks_from_milvus(collection, file_id)
        # Delete from PG
        chunk_repo = KnowledgeChunkRepository()
        await chunk_repo.delete_by_file_id(file_id)
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
                {"id": c.chunk_id, "content": c.content, "chunk_index": c.chunk_index}
                for c in chunks
            ]
        }

    async def get_file_info(self, kb_id: str, file_id: str) -> dict:
        basic = await self.get_file_basic_info(kb_id, file_id)
        content = await self.get_file_content(kb_id, file_id)
        return {**basic, **content}

    async def delete_database(self, kb_id: str) -> dict:
        try:
            if self._connected and utility.has_collection(kb_id, using=self.connection_alias):
                utility.drop_collection(kb_id, using=self.connection_alias)
        except Exception as e:
            logger.error(f"Failed to drop collection {kb_id}: {e}")
        return await super().delete_database(kb_id)


def _count_tokens(text: str) -> int:
    """Approximate token count for statistics."""
    import re
    if not text:
        return 0
    parts = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)
    return max(1, len(parts)) if text.strip() else 0


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
