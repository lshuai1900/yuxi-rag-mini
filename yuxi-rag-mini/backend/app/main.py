import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import logger
from app.rag.storage.database import init_db
from app.rag.manager import KnowledgeBaseManager
from app.rag.factory import KnowledgeBaseFactory
from app.rag.backends.milvus_kb import MilvusKB
from app.api import kb_routes, file_routes, query_routes, chat_routes, health_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.APP_NAME}...")
    logger.info(f"Configuration:\n{settings.log_config()}")
    await init_db()

    # Register MilvusKB
    KnowledgeBaseFactory.register(MilvusKB)

    # Initialize manager
    work_dir = os.path.join(os.getcwd(), "data", "kb_work")
    manager = KnowledgeBaseManager(work_dir)
    await manager.initialize()
    kb_routes.set_manager(manager)

    logger.info(f"{settings.APP_NAME} started successfully")
    yield

    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_routes.router)
app.include_router(kb_routes.router)
app.include_router(file_routes.router)
app.include_router(query_routes.router)
app.include_router(chat_routes.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
