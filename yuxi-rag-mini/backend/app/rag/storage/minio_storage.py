import asyncio
from typing import Any

from app.core.logging import logger
from app.core.config import settings


class MinIOStorage:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from minio import Minio
            self._client = Minio(
                endpoint=settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
            bucket = settings.MINIO_BUCKET
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
                logger.info(f"Created MinIO bucket: {bucket}")
        return self._client

    async def upload_file(self, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        import io
        client = self._get_client()
        bucket = settings.MINIO_BUCKET
        await asyncio.to_thread(
            client.put_object,
            bucket, object_name, io.BytesIO(data), len(data),
            content_type=content_type,
        )
        return f"minio://{bucket}/{object_name}"

    async def download_file(self, object_name: str) -> bytes:
        client = self._get_client()
        bucket = settings.MINIO_BUCKET
        response = await asyncio.to_thread(client.get_object, bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    async def delete_file(self, object_name: str) -> None:
        client = self._get_client()
        bucket = settings.MINIO_BUCKET
        await asyncio.to_thread(client.remove_object, bucket, object_name)

    async def file_exists(self, object_name: str) -> bool:
        client = self._get_client()
        bucket = settings.MINIO_BUCKET
        try:
            await asyncio.to_thread(client.stat_object, bucket, object_name)
            return True
        except Exception:
            return False
