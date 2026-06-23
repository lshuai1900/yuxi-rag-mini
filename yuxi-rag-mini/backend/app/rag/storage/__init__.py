from app.core.config import settings


def get_file_storage():
    if settings.STORAGE_TYPE == "minio":
        from app.rag.storage.minio_storage import MinIOStorage
        return MinIOStorage()
    else:
        from app.rag.storage.local_file_storage import LocalFileStorage
        return LocalFileStorage()
