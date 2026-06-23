from sqlalchemy import Column, Integer, String, Text, BigInteger, Boolean, DateTime, JSON
from sqlalchemy.sql import func
from app.rag.storage.database import Base


class KnowledgeBaseModel(Base):
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    kb_id = Column(String(80), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    kb_type = Column(String(32), nullable=False, index=True)
    embedding_model_spec = Column(String(512))
    additional_params = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class KnowledgeFileModel(Base):
    __tablename__ = "knowledge_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String(64), unique=True, nullable=False, index=True)
    kb_id = Column(String(80), nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    original_filename = Column(String(512))
    file_type = Column(String(64))
    path = Column(String(1024))
    markdown_file = Column(String(1024))
    status = Column(String(32), default="uploaded", index=True)
    content_hash = Column(String(128), index=True)
    file_size = Column(BigInteger)
    chunk_count = Column(Integer, default=0)
    token_count = Column(BigInteger, default=0)
    content_type = Column(String(64))
    processing_params = Column(JSON)
    is_folder = Column(Boolean, default=False)
    error_message = Column(Text)
    failed_reason = Column(Text)
    failed_stage = Column(String(32))
    failed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class KnowledgeChunkModel(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(128), nullable=False, index=True)
    file_id = Column(String(64), nullable=False, index=True)
    kb_id = Column(String(80), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    start_char_pos = Column(Integer)
    end_char_pos = Column(Integer)
    start_token_pos = Column(Integer)
    end_token_pos = Column(Integer)
    graph_indexed = Column(Boolean, default=False, index=True)
    ent_ids = Column(JSON)
    tags = Column(JSON)
    extraction_result = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
