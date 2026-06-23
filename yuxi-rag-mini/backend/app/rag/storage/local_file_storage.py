import os
import aiofiles
from app.core.logging import logger
from app.core.config import settings


class LocalFileStorage:
    def __init__(self, base_path: str | None = None):
        self.base_path = base_path or settings.LOCAL_STORAGE_PATH
        os.makedirs(self.base_path, exist_ok=True)

    async def upload_file(self, object_name: str, data: bytes, content_type: str = "") -> str:
        file_path = os.path.join(self.base_path, object_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(data)
        return f"local://{object_name}"

    async def download_file(self, object_name: str) -> bytes:
        file_path = os.path.join(self.base_path, object_name)
        async with aiofiles.open(file_path, "rb") as f:
            return await f.read()

    async def delete_file(self, object_name: str) -> None:
        file_path = os.path.join(self.base_path, object_name)
        if os.path.exists(file_path):
            os.remove(file_path)

    async def file_exists(self, object_name: str) -> bool:
        file_path = os.path.join(self.base_path, object_name)
        return os.path.exists(file_path)

    def get_local_path(self, object_name: str) -> str:
        return os.path.join(self.base_path, object_name)
