import os
from typing import Any

from app.rag.base import KBNotFoundError, KnowledgeBase
from app.rag.factory import KnowledgeBaseFactory
from app.core.logging import logger


class KnowledgeBaseManager:
    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.kb_instances: dict[str, KnowledgeBase] = {}

    async def initialize(self):
        await self._initialize_existing_kbs()
        logger.info("KnowledgeBaseManager initialized")

    async def _initialize_existing_kbs(self):
        from app.rag.repositories.kb_repository import KnowledgeBaseRepository
        kb_repo = KnowledgeBaseRepository()
        rows = await kb_repo.get_all()
        kb_types_in_use = set()
        for row in rows:
            kb_type = row.kb_type or "milvus"
            if KnowledgeBaseFactory.is_type_supported(kb_type):
                kb_types_in_use.add(kb_type)
        for kb_type in kb_types_in_use:
            try:
                kb_instance = self._get_or_create_kb_instance(kb_type)
                await kb_instance._load_metadata()
                logger.info(f"Initialized {kb_type} knowledge base")
            except Exception as e:
                logger.error(f"Failed to initialize {kb_type}: {e}")

    def _get_or_create_kb_instance(self, kb_type: str) -> KnowledgeBase:
        if kb_type in self.kb_instances:
            return self.kb_instances[kb_type]
        kb_work_dir = os.path.join(self.work_dir, f"{kb_type}_data")
        kb_instance = KnowledgeBaseFactory.create(kb_type, kb_work_dir)
        self.kb_instances[kb_type] = kb_instance
        return kb_instance

    async def _get_kb_for_database(self, kb_id: str) -> KnowledgeBase:
        from app.rag.repositories.kb_repository import KnowledgeBaseRepository
        kb_repo = KnowledgeBaseRepository()
        kb = await kb_repo.get_by_kb_id(kb_id)
        if kb is None:
            raise KBNotFoundError(f"Database {kb_id} not found")
        kb_type = kb.kb_type or "milvus"
        if not KnowledgeBaseFactory.is_type_supported(kb_type):
            raise KBNotFoundError(f"Unsupported kb type: {kb_type}")
        return self._get_or_create_kb_instance(kb_type)

    async def create_database(self, name: str, description: str = "", kb_type: str = "milvus",
                              embedding_model_spec: str | None = None, **kwargs) -> dict:
        if not KnowledgeBaseFactory.is_type_supported(kb_type):
            raise ValueError(f"Unsupported kb type: {kb_type}")
        kb_instance = self._get_or_create_kb_instance(kb_type)
        return await kb_instance.create_database(name, description, embedding_model_spec, **kwargs)

    async def delete_database(self, kb_id: str) -> dict:
        kb_instance = await self._get_kb_for_database(kb_id)
        result = await kb_instance.delete_database(kb_id)
        from app.rag.repositories.kb_repository import KnowledgeBaseRepository
        await KnowledgeBaseRepository().delete(kb_id)
        return result

    async def get_databases(self) -> dict:
        from app.rag.repositories.kb_repository import KnowledgeBaseRepository
        kb_repo = KnowledgeBaseRepository()
        rows = await kb_repo.get_all()
        all_databases = []
        for row in rows:
            kb_type = row.kb_type or "milvus"
            if not KnowledgeBaseFactory.is_type_supported(kb_type):
                continue
            kb_instance = self._get_or_create_kb_instance(kb_type)
            db_info = kb_instance.get_database_info(row.kb_id, include_files=False)
            if db_info:
                all_databases.append(db_info)
        return {"databases": all_databases}

    async def get_database_info(self, kb_id: str, include_files: bool = False) -> dict | None:
        kb_instance = await self._get_kb_for_database(kb_id)
        return kb_instance.get_database_info(kb_id, include_files=include_files)

    async def add_file_record(self, kb_id: str, item: str, params: dict | None = None) -> dict:
        kb_instance = await self._get_kb_for_database(kb_id)
        return await kb_instance.add_file_record(kb_id, item, params)

    async def parse_file(self, kb_id: str, file_id: str) -> dict:
        kb_instance = await self._get_kb_for_database(kb_id)
        return await kb_instance.parse_file(kb_id, file_id)

    async def index_file(self, kb_id: str, file_id: str) -> dict:
        kb_instance = await self._get_kb_for_database(kb_id)
        return await kb_instance.index_file(kb_id, file_id)

    async def aquery(self, query_text: str, kb_id: str, **kwargs) -> list[dict]:
        kb_instance = await self._get_kb_for_database(kb_id)
        return await kb_instance.aquery(query_text, kb_id, **kwargs)

    async def delete_file(self, kb_id: str, file_id: str) -> None:
        kb_instance = await self._get_kb_for_database(kb_id)
        await kb_instance.delete_file(kb_id, file_id)
