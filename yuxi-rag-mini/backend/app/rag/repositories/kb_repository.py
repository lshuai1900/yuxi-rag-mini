from typing import Any
from sqlalchemy import select
from app.rag.storage.database import async_session
from app.rag.storage.models import KnowledgeBaseModel


class KnowledgeBaseRepository:
    async def get_all(self) -> list[KnowledgeBaseModel]:
        async with async_session() as session:
            result = await session.execute(select(KnowledgeBaseModel))
            return list(result.scalars().all())

    async def get_by_kb_id(self, kb_id: str) -> KnowledgeBaseModel | None:
        async with async_session() as session:
            result = await session.execute(select(KnowledgeBaseModel).where(KnowledgeBaseModel.kb_id == kb_id))
            return result.scalar_one_or_none()

    async def create(self, data: dict[str, Any]) -> KnowledgeBaseModel:
        kb = KnowledgeBaseModel(**data)
        async with async_session() as session:
            session.add(kb)
            await session.commit()
        return kb

    async def update(self, kb_id: str, data: dict[str, Any]) -> KnowledgeBaseModel | None:
        async with async_session() as session:
            result = await session.execute(select(KnowledgeBaseModel).where(KnowledgeBaseModel.kb_id == kb_id))
            kb = result.scalar_one_or_none()
            if kb is None:
                return None
            for key, value in data.items():
                if hasattr(kb, key):
                    setattr(kb, key, value)
            await session.commit()
        return kb

    async def delete(self, kb_id: str) -> None:
        async with async_session() as session:
            result = await session.execute(select(KnowledgeBaseModel).where(KnowledgeBaseModel.kb_id == kb_id))
            kb = result.scalar_one_or_none()
            if kb is not None:
                await session.delete(kb)
                await session.commit()
