from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings
from app.core.logging import logger


class Base(DeclarativeBase):
    pass


def get_database_url() -> str:
    if settings.DB_TYPE == "postgresql" and settings.DATABASE_URL:
        return settings.DATABASE_URL
    return f"sqlite+aiosqlite:///{settings.SQLITE_PATH}"


engine = create_async_engine(get_database_url(), echo=settings.DEBUG)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session


async def init_db():
    import os
    from app.rag.storage import models  # noqa: F401 — ensure models are registered with Base
    if settings.DB_TYPE == "sqlite":
        os.makedirs(os.path.dirname(settings.SQLITE_PATH), exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info(f"Database initialized: {get_database_url()}")
