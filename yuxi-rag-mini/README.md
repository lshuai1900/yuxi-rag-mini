# Yuxi RAG Mini

A lightweight RAG (Retrieval-Augmented Generation) knowledge base system, extracted and refactored from the [Yuxi](https://github.com/xerrors/Yuxi) project.

## Features

- **Knowledge Base Management**: Create, list, delete knowledge bases
- **File Upload & Parsing**: Support PDF, Word (.docx), Markdown (.md), TXT
- **Chunking**: Token-aware text chunking with markdown heading support
- **Vector Search**: Milvus-powered vector similarity search
- **BM25 Keyword Search**: Full-text search using Milvus BM25
- **Hybrid Search**: Weighted fusion of vector + BM25 results
- **Pluggable Embedding**: OpenAI-compatible, Ollama, HuggingFace providers
- **Pluggable Storage**: Local filesystem or MinIO
- **Pluggable Database**: SQLite (default) or PostgreSQL
- **GraphRAG Interface**: Placeholder for future graph retrieval

## Architecture

```
yuxi-rag-mini/
├── backend/
│   └── app/
│       ├── main.py              # FastAPI entry point
│       ├── api/                 # REST API routes
│       ├── core/                # Config, logging, errors
│       └── rag/
│           ├── base.py          # KnowledgeBase ABC
│           ├── factory.py       # KB factory
│           ├── manager.py       # KB manager
│           ├── schemas.py       # Pydantic schemas
│           ├── backends/        # MilvusKB implementation
│           ├── chunking/        # Text & markdown chunkers
│           ├── parser/          # PDF, DOCX, MD, TXT parsers
│           ├── providers/       # Embedding & rerank providers
│           ├── storage/         # DB models, file storage, Milvus client
│           ├── repositories/    # DB repositories
│           └── graphrag/        # GraphRAG interfaces (placeholder)
└── frontend/
    └── src/                    # Vue 3 + Vite frontend
```

## Quick Start

### 1. Start Milvus

```bash
docker compose up -d milvus
# Wait for Milvus to be healthy
docker compose logs -f milvus
```

### 2. Start Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -e ..

# Copy and edit environment
cp ../.env.example .env
# Edit .env to configure embedding provider, etc.

# Run server
python -m app.main
# Or: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Complete RAG Workflow

1. Open http://localhost:3000
2. Create a knowledge base
3. Upload a PDF/Word/Markdown/TXT file
4. Click "Ingest" to parse and index the file
5. Switch to "Query" tab, select search mode, and search

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | Health check |
| GET | /api/kb | List knowledge bases |
| POST | /api/kb | Create knowledge base |
| GET | /api/kb/{kb_id} | Get knowledge base info |
| DELETE | /api/kb/{kb_id} | Delete knowledge base |
| POST | /api/kb/{kb_id}/files/upload | Upload file |
| POST | /api/kb/{kb_id}/files/{file_id}/parse | Parse file |
| POST | /api/kb/{kb_id}/files/{file_id}/index | Index file |
| POST | /api/kb/{kb_id}/files/{file_id}/ingest | Parse + Index in one step |
| GET | /api/kb/{kb_id}/files | List files |
| DELETE | /api/kb/{kb_id}/files/{file_id} | Delete file |
| POST | /api/kb/{kb_id}/query | Query (vector/keyword/hybrid) |

## Configuration

See `.env.example` for all configuration options.

### Embedding Providers

| Provider | EMBEDDING_PROVIDER | Notes |
|----------|-------------------|-------|
| OpenAI Compatible | `openai_compatible` | Works with any OpenAI-compatible API |
| Ollama | `ollama` | Uses Ollama's OpenAI-compatible endpoint |
| HuggingFace | `huggingface` | Local sentence-transformers model |

### Storage

| Type | STORAGE_TYPE | Notes |
|------|-------------|-------|
| Local | `local` | Default, files stored in `data/files/` |
| MinIO | `minio` | S3-compatible object storage |

### Database

| Type | DB_TYPE | Notes |
|------|---------|-------|
| SQLite | `sqlite` | Default, zero-config |
| PostgreSQL | `postgresql` | Set DATABASE_URL |

## Search Modes

- **Vector**: Semantic similarity search using embeddings
- **Keyword**: BM25 full-text search (no embedding needed)
- **Hybrid**: Weighted fusion of vector + BM25 results

## Credits

This project is extracted and refactored from [Yuxi](https://github.com/xerrors/Yuxi), retaining the core RAG knowledge base design while removing Agent, MCP, user system, and multi-tenant features.
