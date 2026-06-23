from app.core.config import settings
from app.core.logging import logger


class MilvusClient:
    """Singleton Milvus connection manager."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        from pymilvus import connections, db, utility
        self.uri = settings.MILVUS_URI
        self.token = settings.MILVUS_TOKEN
        self.milvus_db = settings.MILVUS_DB
        self.alias = "default"
        connections.connect(alias=self.alias, uri=self.uri, token=self.token)
        try:
            if self.milvus_db not in db.list_database():
                db.create_database(self.milvus_db)
            db.using_database(self.milvus_db)
        except Exception as e:
            logger.warning(f"Database operation failed, using default: {e}")
        self._initialized = True
        logger.info(f"Connected to Milvus at {self.uri}")
