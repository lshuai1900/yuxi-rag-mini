from typing import Any
from sqlalchemy import select, delete, func, or_
from app.rag.storage.database import async_session
from app.rag.storage.models import KnowledgeChunkModel


class KnowledgeChunkRepository:
    async def get_all(self) -> list[KnowledgeChunkModel]:
        async with async_session() as session:
            result = await session.execute(select(KnowledgeChunkModel))
            return list(result.scalars().all())

    async def list_by_file_id(self, file_id: str) -> list[KnowledgeChunkModel]:
        async with async_session() as session:
            result = await session.execute(
                select(KnowledgeChunkModel)
                .where(KnowledgeChunkModel.file_id == file_id)
                .order_by(KnowledgeChunkModel.chunk_index.asc())
            )
            return list(result.scalars().all())

    async def list_by_kb_id(self, kb_id: str) -> list[KnowledgeChunkModel]:
        async with async_session() as session:
            result = await session.execute(
                select(KnowledgeChunkModel)
                .where(KnowledgeChunkModel.kb_id == kb_id)
                .order_by(KnowledgeChunkModel.chunk_index.asc())
            )
            return list(result.scalars().all())

    async def batch_upsert(self, chunks: list[dict[str, Any]]) -> list[KnowledgeChunkModel]:
        if not chunks:
            return []
        async with async_session() as session:
            chunk_ids = [c["chunk_id"] for c in chunks]
            existing_map = {}
            if chunk_ids:
                result = await session.execute(select(KnowledgeChunkModel).where(KnowledgeChunkModel.chunk_id.in_(chunk_ids)))
                existing_map = {c.chunk_id: c for c in result.scalars().all()}
            records = []
            for chunk_data in chunks:
                chunk_id = chunk_data["chunk_id"]
                record = existing_map.get(chunk_id)
                if record is None:
                    record = KnowledgeChunkModel(**chunk_data)
                    session.add(record)
                else:
                    for key, value in chunk_data.items():
                        if hasattr(record, key):
                            setattr(record, key, value)
                records.append(record)
            await session.commit()
        return records

    async def delete_by_file_id(self, file_id: str) -> int:
        async with async_session() as session:
            result = await session.execute(delete(KnowledgeChunkModel).where(KnowledgeChunkModel.file_id == file_id))
            await session.commit()
            return int(result.rowcount or 0)

    async def delete_by_kb_id(self, kb_id: str) -> int:
        async with async_session() as session:
            result = await session.execute(delete(KnowledgeChunkModel).where(KnowledgeChunkModel.kb_id == kb_id))
            await session.commit()
            return int(result.rowcount or 0)

    async def count_by_file_ids(self, file_ids: list[str]) -> dict[str, int]:
        if not file_ids:
            return {}
        async with async_session() as session:
            result = await session.execute(
                select(KnowledgeChunkModel.file_id, func.count())
                .where(KnowledgeChunkModel.file_id.in_(file_ids))
                .group_by(KnowledgeChunkModel.file_id)
            )
            return {str(fid): int(cnt or 0) for fid, cnt in result.all()}

    async def search_by_keyword(self, kb_id: str, query: str, limit: int = 10) -> list[KnowledgeChunkModel]:
        """Simple keyword search using SQL LIKE. Fallback for BM25."""
        if not query or not query.strip():
            return []
        keywords = [kw.strip() for kw in query.strip().split() if kw.strip()]
        if not keywords:
            return []
        async with async_session() as session:
            conditions = []
            for kw in keywords:
                conditions.append(KnowledgeChunkModel.content.ilike(f"%{kw}%"))
            stmt = (
                select(KnowledgeChunkModel)
                .where(KnowledgeChunkModel.kb_id == kb_id)
                .where(or_(*conditions))
                .order_by(KnowledgeChunkModel.chunk_index.asc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())
