import os
import hashlib
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.api.kb_routes import get_manager
from app.rag.storage import get_file_storage
from app.rag.schemas import IndexResponse, ErrorDetail
from app.rag.base import FileStatus
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
        raise HTTPException(status_code=400, detail=ErrorDetail(
            code="UPLOAD_FAILED",
            message=str(e),
            details={"kb_id": kb_id, "filename": file.filename},
        ).model_dump())


@router.post("/{file_id}/parse")
async def parse_file(kb_id: str, file_id: str):
    manager = get_manager()
    try:
        result = await manager.parse_file(kb_id, file_id)
        return result
    except Exception as e:
        logger.error(f"Parse failed: {e}")
        raise HTTPException(status_code=400, detail=ErrorDetail(
            code="PARSE_FAILED",
            message=str(e),
            details={"kb_id": kb_id, "file_id": file_id},
        ).model_dump())


@router.post("/{file_id}/index", response_model=IndexResponse)
async def index_file(kb_id: str, file_id: str):
    """Parse and index a file: parse -> chunk -> embed -> write to Milvus + SQLite."""
    manager = get_manager()
    try:
        # Auto-parse if not yet parsed
        kb_instance = await manager._get_kb_for_database(kb_id)
        if kb_instance and file_id in kb_instance.files_meta:
            file_meta = kb_instance.files_meta[file_id]
            if file_meta.get("status") in [FileStatus.UPLOADED]:
                await manager.parse_file(kb_id, file_id)
            elif file_meta.get("status") in [FileStatus.ERROR_PARSING, FileStatus.ERROR_INDEXING]:
                # Re-try from appropriate stage
                if file_meta.get("status") == FileStatus.ERROR_PARSING:
                    await manager.parse_file(kb_id, file_id)

        result = await manager.index_file(kb_id, file_id)
        return IndexResponse(**result)
    except RuntimeError as e:
        logger.error(f"Index failed: {e}")
        raise HTTPException(status_code=503, detail=ErrorDetail(
            code="SERVICE_UNAVAILABLE",
            message=str(e),
            details={"kb_id": kb_id, "file_id": file_id},
        ).model_dump())
    except Exception as e:
        logger.error(f"Index failed: {e}")
        raise HTTPException(status_code=400, detail=ErrorDetail(
            code="INDEX_FAILED",
            message=str(e),
            details={"kb_id": kb_id, "file_id": file_id},
        ).model_dump())


@router.post("/{file_id}/ingest", response_model=IndexResponse)
async def ingest_file(kb_id: str, file_id: str):
    """Parse and index a file in one step."""
    manager = get_manager()
    try:
        await manager.parse_file(kb_id, file_id)
        result = await manager.index_file(kb_id, file_id)
        return IndexResponse(**result)
    except RuntimeError as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=503, detail=ErrorDetail(
            code="SERVICE_UNAVAILABLE",
            message=str(e),
            details={"kb_id": kb_id, "file_id": file_id},
        ).model_dump())
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=400, detail=ErrorDetail(
            code="INGEST_FAILED",
            message=str(e),
            details={"kb_id": kb_id, "file_id": file_id},
        ).model_dump())


@router.get("")
async def list_files(kb_id: str):
    manager = get_manager()
    db_info = await manager.get_database_info(kb_id, include_files=True)
    if db_info is None:
        raise HTTPException(status_code=404, detail=ErrorDetail(
            code="KB_NOT_FOUND",
            message=f"Knowledge base {kb_id} not found",
            details={"kb_id": kb_id},
        ).model_dump())
    return db_info.get("files", {})


@router.delete("/{file_id}")
async def delete_file(kb_id: str, file_id: str):
    manager = get_manager()
    try:
        await manager.delete_file(kb_id, file_id)
        return {"message": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=ErrorDetail(
            code="DELETE_FAILED",
            message=str(e),
            details={"kb_id": kb_id, "file_id": file_id},
        ).model_dump())
