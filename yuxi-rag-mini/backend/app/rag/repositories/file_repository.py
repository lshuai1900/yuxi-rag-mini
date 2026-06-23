from typing import Any
from sqlalchemy import select
from app.rag.storage.database import async_session
from app.rag.storage.models import KnowledgeFileModel


class KnowledgeFileRepository:
    async def get_all(self) -> list[KnowledgeFileModel]:
        async with async_session() as session:
            result = await session.execute(select(KnowledgeFileModel))
            return list(result.scalars().all())

    async def get_by_file_id(self, file_id: str) -> KnowledgeFileModel | None:
        async with async_session() as session:
            result = await session.execute(select(KnowledgeFileModel).where(KnowledgeFileModel.file_id == file_id))
            return result.scalar_one_or_none()

    async def list_by_kb_id(self, kb_id: str) -> list[KnowledgeFileModel]:
        async with async_session() as session:
            result = await session.execute(select(KnowledgeFileModel).where(KnowledgeFileModel.kb_id == kb_id))
            return list(result.scalars().all())

    async def upsert(self, file_id: str, data: dict[str, Any]) -> KnowledgeFileModel:
        async with async_session() as session:
            result = await session.execute(select(KnowledgeFileModel).where(KnowledgeFileModel.file_id == file_id))
            existing = result.scalar_one_or_none()
            if existing is None:
                record = KnowledgeFileModel(file_id=file_id, **data)
                session.add(record)
            else:
                for key, value in data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
            await session.commit()
            return existing or record

    async def delete(self, file_id: str) -> None:
        async with async_session() as session:
            result = await session.execute(select(KnowledgeFileModel).where(KnowledgeFileModel.file_id == file_id))
            record = result.scalar_one_or_none()
            if record is not None:
                await session.delete(record)
                await session.commit()

    async def delete_by_kb_id(self, kb_id: str) -> None:
        async with async_session() as session:
            result = await session.execute(select(KnowledgeFileModel).where(KnowledgeFileModel.kb_id == kb_id))
            for record in result.scalars().all():
                await session.delete(record)
            await session.commit()
