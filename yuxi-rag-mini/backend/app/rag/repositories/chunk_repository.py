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

    async def search_by_keyword(self, kb_id: str, query: str, limit: int = 10) -> list[tuple[KnowledgeChunkModel, float]]:
        """Keyword search using SQLite LIKE with multi-keyword matching and scoring.

        Scoring is based on:
        - Number of query keywords that match in the content
        - Slight normalization by content length

        Returns list of (chunk_model, score) tuples.
        Keyword search currently uses SQLite fallback. Milvus BM25 is reserved for future enhancement.
        """
        if not query or not query.strip():
            return []
        keywords = [kw.strip().lower() for kw in query.strip().split() if kw.strip()]
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
                .limit(limit * 3)  # Fetch more for better scoring
            )
            result = await session.execute(stmt)
            chunks = list(result.scalars().all())

        # Score each chunk based on keyword hit count and content length
        scored = []
        for chunk in chunks:
            content_lower = chunk.content.lower()
            hit_count = sum(1 for kw in keywords if kw in content_lower)
            if hit_count == 0:
                continue
            # Base score: ratio of matched keywords to total keywords
            base_score = hit_count / len(keywords)
            # Slight length normalization: shorter content with same hits is slightly more relevant
            content_len = len(chunk.content)
            length_factor = 1.0 / (1.0 + (content_len / 5000.0))
            score = base_score * (0.8 + 0.2 * length_factor)
            scored.append((chunk, round(score, 4)))

        # Sort by score descending, then by chunk_index for stability
        scored.sort(key=lambda x: (-x[1], x[0].chunk_index))
        return scored[:limit]
