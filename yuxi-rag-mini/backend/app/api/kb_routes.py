from fastapi import APIRouter, HTTPException
from app.rag.schemas import KBCreateSchema, KBInfoSchema
from app.rag.manager import KnowledgeBaseManager

router = APIRouter(prefix="/api/kb", tags=["Knowledge Base"])

_manager: KnowledgeBaseManager | None = None


def get_manager() -> KnowledgeBaseManager:
    global _manager
    if _manager is None:
        raise RuntimeError("KnowledgeBaseManager not initialized")
    return _manager


def set_manager(manager: KnowledgeBaseManager):
    global _manager
    _manager = manager


@router.get("", response_model=dict)
async def list_knowledge_bases():
    manager = get_manager()
    return await manager.get_databases()


@router.post("", response_model=dict)
async def create_knowledge_base(body: KBCreateSchema):
    manager = get_manager()
    try:
        result = await manager.create_database(
            name=body.name,
            description=body.description,
            embedding_model_spec=body.embedding_model_spec,
            additional_params=body.additional_params,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{kb_id}", response_model=dict)
async def get_knowledge_base(kb_id: str, include_files: bool = False):
    manager = get_manager()
    result = await manager.get_database_info(kb_id, include_files=include_files)
    if result is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return result


@router.delete("/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    manager = get_manager()
    try:
        return await manager.delete_database(kb_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
