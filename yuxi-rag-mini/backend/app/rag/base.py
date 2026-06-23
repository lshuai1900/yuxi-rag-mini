import asyncio
import os
import secrets
import string
from abc import ABC, abstractmethod
from typing import Any

from app.core.logging import logger
from app.rag.schemas import SearchOutputSchema, SearchResultSchema


class FileStatus:
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    ERROR_PARSING = "error_parsing"
    INDEXING = "indexing"
    INDEXED = "indexed"
    ERROR_INDEXING = "error_indexing"


class KnowledgeBaseException(Exception):
    pass


class KBNotFoundError(KnowledgeBaseException):
    pass


class KnowledgeBase(ABC):
    kb_type = ""
    name = ""
    description = ""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.databases_meta: dict[str, dict] = {}
        self.files_meta: dict[str, dict] = {}
        os.makedirs(work_dir, exist_ok=True)

    @abstractmethod
    async def _create_kb_instance(self, kb_id: str, config: dict) -> Any:
        pass

    @abstractmethod
    async def _initialize_kb_instance(self, instance: Any) -> None:
        pass

    @abstractmethod
    async def index_file(self, kb_id: str, file_id: str, operator_id: str | None = None) -> dict:
        pass

    @abstractmethod
    async def aquery(self, query_text: str, kb_id: str, **kwargs) -> list[dict]:
        pass

    @abstractmethod
    async def delete_file(self, kb_id: str, file_id: str) -> None:
        pass

    @abstractmethod
    async def get_file_basic_info(self, kb_id: str, file_id: str) -> dict:
        pass

    @abstractmethod
    async def get_file_content(self, kb_id: str, file_id: str) -> dict:
        pass

    @abstractmethod
    async def get_file_info(self, kb_id: str, file_id: str) -> dict:
        pass

    async def create_database(
        self,
        database_name: str,
        description: str,
        embedding_model_spec: str | None = None,
        **kwargs,
    ) -> dict:
        kwargs["stats"] = {"file_count": 0, "chunk_count": 0, "token_count": 0}
        alphabet = string.ascii_lowercase + string.digits
        while True:
            kb_id = "kb_" + "".join(secrets.choice(alphabet) for _ in range(10))
            if kb_id not in self.databases_meta:
                break

        self.databases_meta[kb_id] = {
            "name": database_name,
            "description": description,
            "kb_type": self.kb_type,
            "embedding_model_spec": embedding_model_spec,
            "metadata": kwargs,
            "created_at": _utc_now(),
        }
        await self._persist_kb(kb_id)

        working_dir = os.path.join(self.work_dir, kb_id)
        os.makedirs(working_dir, exist_ok=True)

        db_dict = self.databases_meta[kb_id].copy()
        db_dict["kb_id"] = kb_id
        db_dict["files"] = {}
        return db_dict

    async def delete_database(self, kb_id: str) -> dict:
        if kb_id in self.databases_meta:
            files_to_delete = [fid for fid, finfo in self.files_meta.items() if finfo.get("kb_id") == kb_id]
            for file_id in files_to_delete:
                del self.files_meta[file_id]
            del self.databases_meta[kb_id]
            await self._save_metadata()
        return {"message": "deleted"}

    def get_database_info(self, kb_id: str, include_files: bool = True) -> dict | None:
        if kb_id not in self.databases_meta:
            return None
        meta = self.databases_meta[kb_id].copy()
        meta["kb_id"] = kb_id
        meta["stats"] = self._build_database_stats(kb_id)
        if include_files:
            db_files = {}
            for file_id, file_info in self.files_meta.items():
                if file_info.get("kb_id") == kb_id:
                    db_files[file_id] = {
                        "file_id": file_id,
                        "filename": file_info.get("filename", ""),
                        "status": file_info.get("status", "uploaded"),
                        "chunk_count": int(file_info.get("chunk_count") or 0),
                        "token_count": int(file_info.get("token_count") or 0),
                    }
            meta["files"] = db_files
        return meta

    def get_databases(self) -> dict:
        databases = []
        for kb_id, meta in self.databases_meta.items():
            db_dict = meta.copy()
            db_dict["kb_id"] = kb_id
            db_dict["stats"] = self._build_database_stats(kb_id)
            databases.append(db_dict)
        return {"databases": databases}

    def _build_database_stats(self, kb_id: str) -> dict[str, int]:
        stats = {"file_count": 0, "chunk_count": 0, "token_count": 0, "total_size": 0}
        for file_info in self.files_meta.values():
            if file_info.get("kb_id") != kb_id:
                continue
            if file_info.get("is_folder"):
                continue
            stats["file_count"] += 1
            stats["chunk_count"] += int(file_info.get("chunk_count") or 0)
            stats["token_count"] += int(file_info.get("token_count") or 0)
            stats["total_size"] += int(file_info.get("size") or 0)
        return stats

    async def refresh_database_stats(self, kb_id: str) -> dict[str, int]:
        stats = self._build_database_stats(kb_id)
        if kb_id in self.databases_meta:
            self.databases_meta[kb_id].setdefault("metadata", {})["stats"] = stats
            await self._persist_kb(kb_id)
        return stats

    @staticmethod
    def build_search_output(kb_id: str, retrieval_results: list[dict]) -> dict:
        if not isinstance(retrieval_results, list):
            return retrieval_results
        results = []
        for index, chunk in enumerate(retrieval_results):
            if not isinstance(chunk, dict):
                continue
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            file_id = metadata.get("file_id") or chunk.get("file_id") or ""
            chunk_id = metadata.get("chunk_id") or chunk.get("chunk_id") or chunk.get("id") or ""
            results.append(
                SearchResultSchema(
                    id=str(chunk_id or f"{file_id}:{index + 1}"),
                    kb_id=str(kb_id),
                    file_id=str(file_id or ""),
                    content=str(chunk.get("content") or ""),
                    metadata=metadata,
                )
            )
        return SearchOutputSchema(kb_id=str(kb_id), results=results).model_dump()

    async def _persist_file(self, file_id: str) -> None:
        from app.rag.repositories.file_repository import KnowledgeFileRepository
        repo = KnowledgeFileRepository()
        if file_id not in self.files_meta:
            return
        meta = self.files_meta[file_id]
        kb_id = meta.get("kb_id")
        if not kb_id:
            return
        await repo.upsert(file_id, {
            "kb_id": kb_id,
            "filename": meta.get("filename") or "",
            "original_filename": meta.get("original_filename"),
            "file_type": meta.get("file_type"),
            "path": meta.get("path"),
            "markdown_file": meta.get("markdown_file"),
            "status": meta.get("status"),
            "content_hash": meta.get("content_hash"),
            "file_size": meta.get("size"),
            "chunk_count": int(meta.get("chunk_count") or 0),
            "token_count": int(meta.get("token_count") or 0),
            "content_type": meta.get("content_type"),
            "processing_params": meta.get("processing_params"),
            "is_folder": meta.get("is_folder", False),
            "error_message": meta.get("error"),
        })

    async def _persist_kb(self, kb_id: str) -> None:
        from app.rag.repositories.kb_repository import KnowledgeBaseRepository
        repo = KnowledgeBaseRepository()
        if kb_id not in self.databases_meta:
            return
        meta = self.databases_meta[kb_id]
        existing = await repo.get_by_kb_id(kb_id)
        payload = {
            "kb_id": kb_id,
            "name": meta.get("name") or kb_id,
            "description": meta.get("description"),
            "kb_type": meta.get("kb_type") or self.kb_type,
            "embedding_model_spec": meta.get("embedding_model_spec"),
            "additional_params": meta.get("metadata") or {},
        }
        if existing is None:
            await repo.create(payload)
        else:
            await repo.update(kb_id, payload)

    async def _save_metadata(self) -> None:
        from app.rag.repositories.kb_repository import KnowledgeBaseRepository
        from app.rag.repositories.file_repository import KnowledgeFileRepository
        kb_repo = KnowledgeBaseRepository()
        file_repo = KnowledgeFileRepository()
        for kb_id, meta in self.databases_meta.items():
            existing = await kb_repo.get_by_kb_id(kb_id)
            payload = {
                "kb_id": kb_id,
                "name": meta.get("name") or kb_id,
                "description": meta.get("description"),
                "kb_type": meta.get("kb_type") or self.kb_type,
                "embedding_model_spec": meta.get("embedding_model_spec"),
                "additional_params": meta.get("metadata") or {},
            }
            if existing is None:
                await kb_repo.create(payload)
        for file_id, meta in self.files_meta.items():
            kb_id = meta.get("kb_id")
            if not kb_id:
                continue
            await file_repo.upsert(file_id, {
                "kb_id": kb_id,
                "filename": meta.get("filename") or "",
                "file_type": meta.get("file_type"),
                "path": meta.get("path"),
                "markdown_file": meta.get("markdown_file"),
                "status": meta.get("status"),
                "content_hash": meta.get("content_hash"),
                "file_size": meta.get("size"),
                "chunk_count": int(meta.get("chunk_count") or 0),
                "token_count": int(meta.get("token_count") or 0),
            })

    async def _load_metadata(self) -> None:
        from app.rag.repositories.kb_repository import KnowledgeBaseRepository
        from app.rag.repositories.file_repository import KnowledgeFileRepository
        kb_repo = KnowledgeBaseRepository()
        file_repo = KnowledgeFileRepository()
        databases = [kb for kb in await kb_repo.get_all() if kb.kb_type == self.kb_type]
        self.databases_meta = {}
        for kb in databases:
            self.databases_meta[kb.kb_id] = {
                "name": kb.name,
                "description": kb.description,
                "kb_type": kb.kb_type,
                "embedding_model_spec": kb.embedding_model_spec,
                "metadata": kb.additional_params or {},
                "created_at": str(kb.created_at) if kb.created_at else _utc_now(),
            }
        self.files_meta = {}
        for kb in databases:
            for record in await file_repo.list_by_kb_id(kb.kb_id):
                self.files_meta[record.file_id] = {
                    "file_id": record.file_id,
                    "kb_id": record.kb_id,
                    "filename": record.filename,
                    "file_type": record.file_type,
                    "path": record.path,
                    "markdown_file": record.markdown_file,
                    "status": record.status,
                    "content_hash": record.content_hash,
                    "size": record.file_size,
                    "chunk_count": int(getattr(record, "chunk_count", 0) or 0),
                    "token_count": int(getattr(record, "token_count", 0) or 0),
                    "content_type": record.content_type,
                    "processing_params": record.processing_params,
                    "is_folder": getattr(record, "is_folder", False),
                    "error": getattr(record, "error_message", None),
                    "original_filename": getattr(record, "original_filename", None),
                }
        logger.info(f"Loaded {self.kb_type} metadata: {len(self.databases_meta)} databases, {len(self.files_meta)} files")


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
