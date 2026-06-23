from app.rag.base import KBNotFoundError, KnowledgeBase
from app.core.logging import logger


class KnowledgeBaseFactory:
    _kb_types: dict[str, type[KnowledgeBase]] = {}

    @classmethod
    def register(cls, kb_class: type[KnowledgeBase]):
        if not issubclass(kb_class, KnowledgeBase):
            raise ValueError("Must inherit from KnowledgeBase")
        if not kb_class.kb_type:
            raise ValueError("Must define kb_type")
        cls._kb_types[kb_class.kb_type] = kb_class

    @classmethod
    def create(cls, kb_type: str, work_dir: str, **kwargs) -> KnowledgeBase:
        if kb_type not in cls._kb_types:
            raise KBNotFoundError(f"Unknown kb type: {kb_type}")
        kb_class = cls._kb_types[kb_type]
        return kb_class(work_dir, **kwargs)

    @classmethod
    def get_available_types(cls) -> dict[str, dict]:
        return {
            kb_type: {"name": kb_class.name, "description": kb_class.description}
            for kb_type, kb_class in cls._kb_types.items()
        }

    @classmethod
    def is_type_supported(cls, kb_type: str) -> bool:
        return kb_type in cls._kb_types
