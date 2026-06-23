# Yuxi RAG Mini

A lightweight RAG (Retrieval-Augmented Generation) knowledge base system, extracted and refactored from the [Yuxi](https://github.com/xerrors/Yuxi) project.

## Features

- **Knowledge Base Management**: Create, list, delete knowledge bases
- **File Upload & Parsing**: Support PDF, Word (.docx), Markdown (.md), TXT
- **Chunking**: Token-aware text chunking with overlap and markdown heading support
- **Vector Search**: Milvus-powered vector similarity search
- **Keyword Search**: SQLite-based keyword matching (fallback for BM25)
- **Hybrid Search**: Weighted fusion of vector + keyword results
- **Pluggable Embedding**: OpenAI-compatible, Ollama, HuggingFace, Fake (test only)
- **Pluggable Storage**: Local filesystem or MinIO
- **Pluggable Database**: SQLite (default) or PostgreSQL
- **GraphRAG Interface**: Placeholder for future graph retrieval
- **Rerank Interface**: DummyReranker placeholder for future reranking

## Architecture

```
yuxi-rag-mini/
├── backend/app/
│   ├── main.py              # FastAPI entry point
│   ├── api/                 # REST API routes
│   │   ├── kb_routes.py     # Knowledge base CRUD
│   │   ├── file_routes.py   # File upload/parse/index
│   │   ├── query_routes.py  # Query (vector/keyword/hybrid)
│   │   └── health_routes.py # Health check
│   ├── core/                # Config, logging, errors
│   └── rag/
│       ├── base.py          # KnowledgeBase ABC
│       ├── factory.py       # KB factory
│       ├── manager.py       # KB manager
│       ├── schemas.py       # Pydantic schemas
│       ├── backends/
│       │   └── milvus_kb.py # MilvusKB implementation
│       ├── chunking/        # Text & markdown chunkers
│       ├── parser/          # PDF, DOCX, MD, TXT parsers
│       ├── providers/
│       │   ├── embedding/   # OpenAI/Ollama/HuggingFace/Fake
│       │   └── rerank/      # DummyReranker + APIReranker
│       ├── storage/         # DB models, file storage, Milvus client
│       ├── repositories/    # DB repositories
│       └── graphrag/        # GraphRAG interfaces (placeholder)
└── frontend/src/            # Vue 3 + Vite frontend
```

## Quick Start

### 1. Start Milvus

**Option A: Milvus Standalone (Docker)**
```bash
docker compose up -d milvus
# Wait for Milvus to be healthy
docker compose logs -f milvus
```

**Option B: Milvus Lite (No Docker, for development)**
Set `MILVUS_URI=data/milvus.db` in `.env`. Milvus Lite runs embedded, no separate server needed.

### 2. Configure Embedding Provider

Copy and edit the environment file:
```bash
cp .env.example .env
```

**For testing (no external API needed):**
```bash
EMBEDDING_PROVIDER=fake
EMBEDDING_DIMENSION=128
```

> **WARNING**: FakeEmbeddingProvider generates meaningless vectors. It should ONLY be used for testing the pipeline. Search results will be meaningless with this provider. For real RAG, use OpenAI-compatible or Ollama.

**For real RAG with Ollama:**
```bash
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=bge-m3
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_DIMENSION=1024
```

**For real RAG with OpenAI-compatible API:**
```bash
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL=bge-m3
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_DIMENSION=1024
```

### 3. Start Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -e ..

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

### 5. Complete RAG Workflow

1. Open http://localhost:3000
2. Create a knowledge base
3. Upload a PDF/Word/Markdown/TXT file
4. Click "Index" to parse, chunk, embed, and store the file
5. Switch to "Query" tab
6. Select search mode (vector/keyword/hybrid)
7. Enter a query and search
8. View results with filename, score, content, and metadata

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
| POST | /api/kb/{kb_id}/files/{file_id}/index | Parse + Index file |
| POST | /api/kb/{kb_id}/files/{file_id}/ingest | Parse + Index in one step |
| GET | /api/kb/{kb_id}/files | List files |
| DELETE | /api/kb/{kb_id}/files/{file_id} | Delete file |
| POST | /api/kb/{kb_id}/query | Query (vector/keyword/hybrid) |

### Query Request Format

```json
{
  "query": "What is deep learning?",
  "search_mode": "hybrid",
  "top_k": 5,
  "similarity_threshold": 0.3,
  "enable_rerank": false,
  "enable_graphrag": false
}
```

### Query Response Format

```json
{
  "query": "What is deep learning?",
  "search_mode": "hybrid",
  "results": [
    {
      "chunk_id": "file_xxx_chunk_0",
      "file_id": "file_xxx",
      "filename": "ai_overview.md",
      "content": "Deep learning uses neural networks...",
      "score": 0.82,
      "metadata": {"page": 1}
    }
  ]
}
```

## Search Modes

| Mode | Description | Requires Embedding |
|------|-------------|-------------------|
| **vector** | Semantic similarity search using embeddings | Yes |
| **keyword** | SQL LIKE-based keyword matching | No |
| **hybrid** | Weighted merge of vector + keyword results | Yes |

## Configuration

See `.env.example` for all configuration options.

### Embedding Providers

| Provider | EMBEDDING_PROVIDER | Notes |
|----------|-------------------|-------|
| OpenAI Compatible | `openai_compatible` | Works with any OpenAI-compatible API |
| Ollama | `ollama` | Uses Ollama's OpenAI-compatible endpoint |
| HuggingFace | `huggingface` | Local sentence-transformers model |
| Fake | `fake` | **Test only!** Generates meaningless vectors |

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

## Testing

```bash
cd backend
pip install pytest pytest-asyncio httpx
pytest ../tests/ -v
```

## Current Limitations

- **GraphRAG**: Only interfaces are defined, not implemented. When `enable_graphrag=true`, returns "GraphRAG is reserved but not implemented yet."
- **Rerank**: Only DummyReranker is available. When `enable_rerank=true`, applies random scores for testing.
- **Keyword Search**: Uses SQLite LIKE matching as fallback. Future: Milvus BM25 full-text search.
- **Hybrid Search**: Uses simple weighted merge. Future: Milvus native hybrid_search with WeightedRanker.
- **FakeEmbeddingProvider**: Only for testing. Search results are meaningless with this provider.

## Credits

This project is extracted and refactored from [Yuxi](https://github.com/xerrors/Yuxi), retaining the core RAG knowledge base design while removing Agent, MCP, user system, and multi-tenant features.
