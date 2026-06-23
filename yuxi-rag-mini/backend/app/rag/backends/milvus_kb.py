import asyncio
import json
import math
import os
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    db,
    utility,
)

from app.rag.base import FileStatus, KnowledgeBase
from app.rag.chunking.text_chunker import ChunkResult, chunk_text, count_tokens
from app.rag.chunking.markdown_chunker import chunk_markdown
from app.rag.parser.factory import parse_file
from app.rag.providers.embedding import create_embedding_provider
from app.rag.providers.rerank import create_reranker
from app.rag.repositories.chunk_repository import KnowledgeChunkRepository
from app.rag.storage import get_file_storage
from app.core.config import settings
from app.core.logging import logger

VECTOR_METRIC_TYPE = "COSINE"


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
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=80),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        ]
        schema = CollectionSchema(fields=fields, description=f"KB collection for {kb_id}")
        collection = Collection(name=collection_name, schema=schema, using=self.connection_alias)
        index_params = {"metric_type": VECTOR_METRIC_TYPE, "index_type": "FLAT"}
        collection.create_index("embedding", index_params, index_name="embedding_idx")
        logger.info(f"Created Milvus collection: {collection_name}")
        return collection

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

    def _do_chunking(self, text: str, file_id: str, kb_id: str, filename: str,
                     pages: list[dict] | None = None, parse_metadata: dict | None = None) -> list[ChunkResult]:
        ext = os.path.splitext(filename)[1].lower() if filename else ""
        chunk_size = settings.EMBEDDING_CHUNK_SIZE
        chunk_overlap = settings.EMBEDDING_CHUNK_OVERLAP

        base_metadata = {}
        if parse_metadata:
            base_metadata.update(parse_metadata)
        if pages:
            base_metadata["has_pages"] = True

        if ext in [".md", ".markdown"]:
            chunks = chunk_markdown(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                    file_id=file_id, kb_id=kb_id, filename=filename, metadata=base_metadata)
        else:
            chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                               file_id=file_id, kb_id=kb_id, filename=filename, metadata=base_metadata)

        # If we have page info, try to assign page numbers to chunks
        if pages:
            for chunk in chunks:
                page_num = self._find_page_for_chunk(chunk.content, pages)
                if page_num is not None:
                    chunk.metadata["page_number"] = page_num

        return chunks

    @staticmethod
    def _find_page_for_chunk(chunk_content: str, pages: list[dict]) -> int | None:
        """Find the page number for a chunk by checking which page contains its text."""
        for page in pages:
            page_text = page.get("text", "")
            if page_text and chunk_content[:50] in page_text:
                return page.get("page_number")
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
        if file_id not in self.files_meta:
            raise ValueError(f"File {file_id} not found")
        file_meta = self.files_meta[file_id]

        self.files_meta[file_id]["status"] = FileStatus.PARSING
        await self._persist_file(file_id)

        try:
            file_path = file_meta.get("path")
            storage = get_file_storage()

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
            parse_result = await parse_file(actual_path, filename=filename)

            # Check for empty parse result
            if not parse_result.text or not parse_result.text.strip():
                raise ValueError(f"No extractable content found in {filename}. The file may be empty or contain only non-text content.")

            # Save parsed content
            parsed_object_name = f"{kb_id}/parsed/{file_id}.md"
            await storage.upload_file(parsed_object_name, parse_result.text.encode("utf-8"), "text/markdown")

            self.files_meta[file_id]["status"] = FileStatus.PARSED
            self.files_meta[file_id]["markdown_file"] = parsed_object_name
            self.files_meta[file_id]["parse_metadata"] = {
                "file_type": parse_result.file_type,
                "total_pages": len(parse_result.pages),
                "pages": parse_result.pages,
            }
            await self._persist_file(file_id)
            return self.files_meta[file_id]
        except Exception as e:
            logger.error(f"Failed to parse file {file_id}: {e}")
            self.files_meta[file_id]["status"] = FileStatus.FAILED
            self.files_meta[file_id]["failed_reason"] = str(e)
            self.files_meta[file_id]["failed_stage"] = "parsing"
            self.files_meta[file_id]["failed_at"] = _utc_now()
            await self._persist_file(file_id)
            raise

    async def index_file(self, kb_id: str, file_id: str, operator_id: str | None = None) -> dict:
        if kb_id not in self.databases_meta:
            raise ValueError(f"Database {kb_id} not found")

        if not self._connected:
            raise RuntimeError("Milvus is not connected. Cannot index files.")

        collection = await self._get_milvus_collection(kb_id)
        if not collection:
            raise RuntimeError(f"Failed to get Milvus collection for {kb_id}")

        if file_id not in self.files_meta:
            raise ValueError(f"File {file_id} not found")
        file_meta = self.files_meta[file_id]

        if not file_meta.get("markdown_file"):
            raise ValueError("File has not been parsed yet. Call parse first.")

        # Stage: chunking
        self.files_meta[file_id]["status"] = FileStatus.CHUNKING
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
            parse_metadata = file_meta.get("parse_metadata", {})
            pages = parse_metadata.get("pages", [])

            # Chunk
            chunks = self._do_chunking(text_content, file_id, kb_id, filename,
                                       pages=pages, parse_metadata=parse_metadata)
            logger.info(f"Split {filename} into {len(chunks)} chunks")

            if not chunks:
                raise ValueError(f"No chunks generated from {filename}. The file content may be too short or empty.")

            # Delete existing chunks for this file
            await self._delete_file_chunks(kb_id, file_id, collection)

            # Stage: embedding
            self.files_meta[file_id]["status"] = FileStatus.EMBEDDING
            await self._persist_file(file_id)

            embedding_provider = self._get_embedding_provider()
            texts = [c.content for c in chunks]
            embeddings = await embedding_provider.embed_documents(texts)
            logger.info(f"Embedded {len(texts)} chunks")

            # Stage: indexing (writing to Milvus + SQLite)
            self.files_meta[file_id]["status"] = FileStatus.INDEXING
            await self._persist_file(file_id)

            ids = [c.chunk_id for c in chunks]
            kb_ids = [c.kb_id for c in chunks]
            file_ids = [c.file_id for c in chunks]
            chunk_ids = [c.chunk_id for c in chunks]
            contents = [c.content for c in chunks]
            filenames = [c.filename for c in chunks]
            chunk_indices = [c.chunk_index for c in chunks]
            metadata_jsons = [json.dumps(c.metadata, ensure_ascii=False) for c in chunks]

            entities = [ids, kb_ids, file_ids, chunk_ids, contents, filenames,
                       chunk_indices, metadata_jsons, embeddings]
            await asyncio.to_thread(collection.insert, entities)
            await asyncio.to_thread(collection.flush)
            logger.info(f"Inserted {len(chunks)} chunks into Milvus")

            # Insert to SQLite
            chunk_repo = KnowledgeChunkRepository()
            chunk_records = [
                {
                    "chunk_id": c.chunk_id,
                    "file_id": c.file_id,
                    "kb_id": c.kb_id,
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                }
                for c in chunks
            ]
            await chunk_repo.batch_upsert(chunk_records)

            chunk_stats = {
                "chunk_count": len(chunks),
                "token_count": sum(count_tokens(c.content) for c in chunks),
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
            self.files_meta[file_id]["status"] = FileStatus.FAILED
            self.files_meta[file_id]["failed_reason"] = str(e)
            # Record the stage where failure occurred
            current_status = self.files_meta[file_id].get("status", FileStatus.FAILED)
            if current_status == FileStatus.CHUNKING:
                self.files_meta[file_id]["failed_stage"] = "chunking"
            elif current_status == FileStatus.EMBEDDING:
                self.files_meta[file_id]["failed_stage"] = "embedding"
            else:
                self.files_meta[file_id]["failed_stage"] = "indexing"
            self.files_meta[file_id]["failed_at"] = _utc_now()
            await self._persist_file(file_id)
            raise

    async def _delete_file_chunks(self, kb_id: str, file_id: str, collection: Collection) -> None:
        chunk_repo = KnowledgeChunkRepository()
        await chunk_repo.delete_by_file_id(file_id)
        try:
            expr = f'file_id == "{file_id}"'
            await asyncio.to_thread(collection.delete, expr)
        except Exception as e:
            logger.warning(f"Error deleting chunks from Milvus: {e}")

    async def aquery(self, query_text: str, kb_id: str, **kwargs) -> list[dict]:
        search_mode = str(kwargs.get("search_mode", "vector")).lower()
        if search_mode not in {"vector", "keyword", "hybrid"}:
            search_mode = "vector"
        top_k = int(kwargs.get("top_k", 10))
        similarity_threshold = float(kwargs.get("similarity_threshold", 0.0))

        if search_mode == "vector":
            return await self._vector_search(query_text, kb_id, top_k, similarity_threshold)
        elif search_mode == "keyword":
            return await self._keyword_search(query_text, kb_id, top_k, similarity_threshold)
        elif search_mode == "hybrid":
            hybrid_kwargs = {k: v for k, v in kwargs.items() if k not in ("top_k", "similarity_threshold", "search_mode")}
            return await self._hybrid_search(query_text, kb_id, top_k, similarity_threshold, **hybrid_kwargs)
        return []

    async def _vector_search(self, query_text: str, kb_id: str, top_k: int,
                             similarity_threshold: float) -> list[dict]:
        if not self._connected:
            raise RuntimeError("Milvus is not connected. Cannot perform vector search.")
        collection = await self._get_milvus_collection(kb_id)
        if not collection:
            raise RuntimeError(f"Collection for {kb_id} not found")

        embedding_provider = self._get_embedding_provider()
        query_embedding = await embedding_provider.embed_query(query_text)
        search_params = {"metric_type": VECTOR_METRIC_TYPE, "params": {}}
        output_fields = ["content", "chunk_id", "file_id", "filename", "chunk_index", "metadata_json"]

        results = await asyncio.to_thread(
            collection.search,
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
        )

        retrieved = []
        if results and len(results) > 0:
            for hit in results[0]:
                score = float(hit.distance) if hit.distance is not None else 0.0
                # For FakeEmbeddingProvider, Milvus may return None distance
                # Set a default score based on rank position
                if hit.distance is None:
                    score = round(1.0 - (len(retrieved) * 0.1), 4)
                if similarity_threshold > 0 and score < similarity_threshold:
                    continue
                entity = hit.entity
                metadata = {}
                try:
                    metadata = json.loads(entity.get("metadata_json", "{}"))
                except Exception:
                    pass
                retrieved.append({
                    "chunk_id": entity.get("chunk_id", ""),
                    "file_id": entity.get("file_id", ""),
                    "filename": entity.get("filename", ""),
                    "content": entity.get("content", ""),
                    "score": score,
                    "score_detail": {
                        "vector_score": score,
                        "keyword_score": 0.0,
                        "final_score": score,
                        "source": "vector",
                    },
                    "metadata": metadata,
                })
        return retrieved

    async def _keyword_search(self, query_text: str, kb_id: str, top_k: int,
                              similarity_threshold: float) -> list[dict]:
        """Keyword search using SQLite fallback with multi-keyword matching and scoring.

        Keyword search currently uses SQLite fallback. Milvus BM25 is reserved for future enhancement.
        """
        chunk_repo = KnowledgeChunkRepository()
        chunks_with_scores = await chunk_repo.search_by_keyword(kb_id, query_text, limit=top_k * 2)

        results = []
        for chunk, kw_score in chunks_with_scores:
            results.append({
                "chunk_id": chunk.chunk_id,
                "file_id": chunk.file_id,
                "filename": self._get_filename(chunk.file_id),
                "content": chunk.content,
                "score": kw_score,
                "score_detail": {
                    "vector_score": 0.0,
                    "keyword_score": kw_score,
                    "final_score": kw_score,
                    "source": "keyword",
                },
                "metadata": {"search_mode": "keyword"},
            })

        # Apply similarity threshold
        if similarity_threshold > 0:
            results = [r for r in results if r["score"] >= similarity_threshold]

        return results[:top_k]

    async def _hybrid_search(self, query_text: str, kb_id: str, top_k: int,
                             similarity_threshold: float, **kwargs) -> list[dict]:
        """Hybrid search combining vector and keyword results with weighted scoring.

        Current implementation: merge vector + keyword results with weighted scoring.
        Future: use Milvus native hybrid_search with WeightedRanker when BM25 is supported.
        """
        vector_weight = float(kwargs.get("vector_weight", 0.7))
        keyword_weight = float(kwargs.get("keyword_weight", 0.3))

        vector_results = []
        keyword_results = []

        try:
            vector_results = await self._vector_search(query_text, kb_id, top_k, 0.0)
        except Exception as e:
            logger.warning(f"Vector search failed in hybrid: {e}")

        try:
            keyword_results = await self._keyword_search(query_text, kb_id, top_k, 0.0)
        except Exception as e:
            logger.warning(f"Keyword search failed in hybrid: {e}")

        # Normalize vector scores to [0, 1]
        vector_scores = [r["score"] for r in vector_results]
        if vector_scores:
            max_vec = max(vector_scores)
            min_vec = min(vector_scores)
            # When all scores are equal, treat as if they all have the same top score
            # Normalize to 1.0 by setting min=0 and range=1
            if max_vec == min_vec:
                min_vec = 0.0
                vec_range = max_vec if max_vec > 0 else 1.0
            else:
                vec_range = max_vec - min_vec
        else:
            max_vec = min_vec = 0.0
            vec_range = 1.0

        # Normalize keyword scores to [0, 1]
        keyword_scores = [r["score"] for r in keyword_results]
        if keyword_scores:
            max_kw = max(keyword_scores)
            min_kw = min(keyword_scores)
            if max_kw == min_kw:
                min_kw = 0.0
                kw_range = max_kw if max_kw > 0 else 1.0
            else:
                kw_range = max_kw - min_kw
        else:
            max_kw = min_kw = 0.0
            kw_range = 1.0

        # Merge results: same chunk from both sources gets combined score
        merged: dict[str, dict] = {}

        for r in vector_results:
            key = r["chunk_id"]
            # When all scores are equal (single result), normalize to 1.0
            if vec_range > 0:
                normalized_vec = (r["score"] - min_vec) / vec_range
            else:
                normalized_vec = 1.0 if r["score"] > 0 else 0.0
            merged[key] = {
                **r,
                "score": normalized_vec * vector_weight,
                "score_detail": {
                    "vector_score": round(normalized_vec, 4),
                    "keyword_score": 0.0,
                    "final_score": round(normalized_vec * vector_weight, 4),
                    "source": "hybrid",
                },
            }

        for r in keyword_results:
            key = r["chunk_id"]
            if kw_range > 0:
                normalized_kw = (r["score"] - min_kw) / kw_range
            else:
                normalized_kw = 1.0 if r["score"] > 0 else 0.0
            weighted_kw = normalized_kw * keyword_weight
            if key in merged:
                # Merge: chunk found in both vector and keyword
                merged[key]["score"] += weighted_kw
                merged[key]["score_detail"]["keyword_score"] = round(normalized_kw, 4)
                merged[key]["score_detail"]["final_score"] = round(merged[key]["score"], 4)
            else:
                merged[key] = {
                    **r,
                    "score": weighted_kw,
                    "score_detail": {
                        "vector_score": 0.0,
                        "keyword_score": round(normalized_kw, 4),
                        "final_score": round(weighted_kw, 4),
                        "source": "hybrid",
                    },
                }

        # Sort by final score descending
        results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

        # Apply similarity threshold on final score
        if similarity_threshold > 0:
            results = [r for r in results if r["score"] >= similarity_threshold]

        return results[:top_k]

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

    def __del__(self):
        try:
            if hasattr(self, "connection_alias"):
                connections.disconnect(self.connection_alias)
        except Exception:
            pass


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
