import os
import hashlib
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.rag.manager import KnowledgeBaseManager
from app.api.kb_routes import get_manager
from app.rag.storage import get_file_storage
from app.core.logging import logger

router = APIRouter(prefix="/api/kb/{kb_id}/files", tags=["Files"])


@router.post("/upload")
async def upload_file(kb_id: str, file: UploadFile = File(...)):
    manager = get_manager()
    try:
        storage = get_file_storage()
        file_content = await file.read()
        content_hash = hashlib.sha256(file_content).hexdigest()
        object_name = f"{kb_id}/documents/{file.filename}"
        await storage.upload_file(object_name, file_content)

        result = await manager.add_file_record(kb_id, object_name, {
            "filename": file.filename,
            "size": len(file_content),
            "content_hash": content_hash,
        })
        return result
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{file_id}/parse")
async def parse_file(kb_id: str, file_id: str):
    manager = get_manager()
    try:
        result = await manager.parse_file(kb_id, file_id)
        return result
    except Exception as e:
        logger.error(f"Parse failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{file_id}/index")
async def index_file(kb_id: str, file_id: str):
    manager = get_manager()
    try:
        result = await manager.index_file(kb_id, file_id)
        return result
    except Exception as e:
        logger.error(f"Index failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{file_id}/ingest")
async def ingest_file(kb_id: str, file_id: str):
    """Parse and index a file in one step."""
    manager = get_manager()
    try:
        parse_result = await manager.parse_file(kb_id, file_id)
        index_result = await manager.index_file(kb_id, file_id)
        return index_result
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("")
async def list_files(kb_id: str):
    manager = get_manager()
    db_info = await manager.get_database_info(kb_id, include_files=True)
    if db_info is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return db_info.get("files", {})


@router.delete("/{file_id}")
async def delete_file(kb_id: str, file_id: str):
    manager = get_manager()
    try:
        await manager.delete_file(kb_id, file_id)
        return {"message": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
