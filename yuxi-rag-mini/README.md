# Yuxi RAG Mini

A lightweight, production-ready **Retrieval-Augmented Generation (RAG)** knowledge base system. Built with FastAPI + Vue 3 + Milvus, supporting multiple document formats, pluggable embedding providers, and hybrid search.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Vue 3 + Vite Frontend                  │
│         (Upload / Index / Query with Score Detail)          │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP API
┌────────────────────────────▼────────────────────────────────┐
│                       FastAPI Backend                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ KB Routes │  │File Routes│  │Query Rts │  │Health Routes │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────────┘ │
│       └──────────────┼─────────────┘                          │
│              ┌───────▼────────┐                               │
│              │ KnowledgeBase  │                               │
│              │    Manager     │                               │
│              └───────┬────────┘                               │
│     ┌────────────────┼────────────────┐                      │
│  ┌──▼──────┐  ┌─────▼─────┐  ┌──────▼──────┐               │
│  │ Parser  │  │  Chunker  │  │  Embedding  │               │
│  │(PDF/DOCX│  │(Text/MD)  │  │  Provider   │               │
│  │ MD/TXT) │  │           │  │(OpenAI/Oll │               │
│  └─────────┘  └───────────┘  │ ama/HF/Fake)│               │
│                              └──────┬──────┘                │
│  ┌──────────┐  ┌──────────┐        │                        │
│  │ Reranker │  │ GraphRAG │        │                        │
│  │ (Dummy)  │  │ (Planned)│        │                        │
│  └──────────┘  └──────────┘        │                        │
└─────────────────────────────────────┼────────────────────────┘
                    ┌─────────────────┼─────────────────┐
              ┌─────▼─────┐    ┌──────▼──────┐   ┌──────▼──────┐
              │  Milvus   │    │   SQLite /  │   │Local / MinIO│
              │(Lite/Stdl)│    │ PostgreSQL  │   │  Storage    │
              └───────────┘    └─────────────┘   └─────────────┘
```

## RAG Pipeline

```
Upload → Parse → Chunk → Embed → Index → Retrieve
  │        │        │       │       │        │
  │     PDF/DOCX  Text/MD  OpenAI  Milvus  Vector
  │     MD/TXT    Chunker  Ollama  +SQLite Keyword
  │                       HF/Fake          Hybrid
```

**File Status State Machine:**

```
uploaded → parsing → parsed → chunking → embedding → indexing → indexed
                                                          ↘ failed (with reason & stage)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Pydantic, SQLAlchemy (async) |
| Frontend | Vue 3, Vite, TypeScript |
| Vector DB | Milvus (Lite / Standalone) |
| Metadata DB | SQLite (aiosqlite) / PostgreSQL (asyncpg) |
| File Storage | Local filesystem / MinIO |
| Embedding | OpenAI-compatible, Ollama, HuggingFace, Fake (testing) |
| Document Parsing | PyMuPDF, python-docx, custom Markdown/TXT parsers |

## Search Modes

### 1. Vector Search
Dense vector similarity search using Milvus. Converts query to embedding and finds nearest neighbors by cosine similarity.

### 2. Keyword Search
Keyword search currently uses SQLite fallback with multi-keyword matching and scoring. Milvus BM25 is reserved for future enhancement.

Scoring is based on:
- Number of query keywords that match in the content
- Slight normalization by content length

### 3. Hybrid Search
Combines vector and keyword results with weighted scoring:

```
final_score = vector_weight × normalized_vector_score + keyword_weight × normalized_keyword_score
```

Default weights: `vector_weight=0.7`, `keyword_weight=0.3`

Results include `score_detail`:
```json
{
  "vector_score": 0.82,
  "keyword_score": 0.50,
  "final_score": 0.724,
  "source": "hybrid"
}
```

Same chunk appearing in both vector and keyword results gets merged into one with combined score.

## Quick Start

### Option A: Milvus Lite (Zero Dependencies)

No Docker needed. Uses embedded Milvus + SQLite.

```bash
# Backend
cd backend
pip install -e ..
cp ../.env.example .env
# Edit .env: set MILVUS_URI=data/milvus.db
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

### Option B: Milvus Standalone (Docker)

```bash
# Start Milvus stack
docker compose up -d milvus

# Backend
cd backend
pip install -e ..
cp ../.env.example .env
# Edit .env: set MILVUS_URI=http://localhost:19530
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

## API Examples

### Create Knowledge Base

```bash
curl -X POST http://localhost:8000/api/kb \
  -H "Content-Type: application/json" \
  -d '{"name": "My KB", "description": "Test knowledge base"}'
```

### Upload File

```bash
curl -X POST http://localhost:8000/api/kb/{kb_id}/files/upload \
  -F "file=@document.pdf"
```

### Index File (parse + chunk + embed + write)

```bash
curl -X POST http://localhost:8000/api/kb/{kb_id}/files/{file_id}/index
```

### Query

```bash
curl -X POST http://localhost:8000/api/kb/{kb_id}/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "search_mode": "hybrid",
    "top_k": 10,
    "similarity_threshold": 0.0,
    "enable_rerank": false
  }'
```

**Query Response:**

```json
{
  "query": "What is machine learning?",
  "search_mode": "hybrid",
  "results": [
    {
      "chunk_id": "chunk_abc123",
      "file_id": "file_xyz789",
      "filename": "ai_overview.pdf",
      "content": "Machine learning is a subset of AI...",
      "score": 0.724,
      "score_detail": {
        "vector_score": 0.82,
        "keyword_score": 0.50,
        "final_score": 0.724,
        "source": "hybrid"
      },
      "metadata": {"page_number": 3}
    }
  ],
  "rerank": {
    "reranked": false,
    "reranker": "dummy"
  }
}
```

## Frontend Usage

1. **Knowledge Bases** tab: Create and manage knowledge bases
2. **Upload** tab: Upload files (PDF, DOCX, MD, TXT), view status, trigger indexing
3. **Query** tab: Search with mode selection (vector/keyword/hybrid), adjust top_k and similarity threshold, view score details

## Configuration

Key environment variables (see `.env.example` for full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `fake` | `openai_compatible` / `ollama` / `huggingface` / `fake` |
| `EMBEDDING_MODEL` | `bge-m3` | Model name for embedding |
| `EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | API base URL |
| `EMBEDDING_API_KEY` | `` | API key (not logged at startup) |
| `EMBEDDING_DIMENSION` | `128` | Embedding vector dimension |
| `EMBEDDING_BATCH_SIZE` | `40` | Batch size for embedding |
| `EMBEDDING_TIMEOUT` | `60` | Timeout in seconds |
| `EMBEDDING_CHUNK_SIZE` | `512` | Chunk size in tokens |
| `EMBEDDING_CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `MILVUS_URI` | `data/milvus.db` | Milvus connection URI |
| `DB_TYPE` | `sqlite` | `sqlite` / `postgresql` |
| `STORAGE_TYPE` | `local` | `local` / `minio` |

> **WARNING**: `EMBEDDING_PROVIDER=fake` generates deterministic but meaningless vectors. It should ONLY be used for testing the pipeline, NOT for real RAG. For resume demonstrations, use a real embedding provider (OpenAI-compatible, Ollama, or HuggingFace).

## Testing

```bash
cd backend
python -m pytest ../tests/ -v
```

Tests use FakeEmbeddingProvider + Milvus Lite + SQLite, no external dependencies needed.

## Current Limitations

- **Keyword search** currently uses SQLite fallback with LIKE matching, not Milvus native BM25. Milvus BM25 is reserved for future enhancement.
- **GraphRAG** interface is reserved but not implemented. Querying with `enable_graphrag=true` returns a placeholder message.
- **Rerank** current version provides rerank interface with DummyReranker. Real rerank model can be plugged in later.
- **PDF** only supports extractable text via PyMuPDF. OCR is not enabled in this version.
- **No user system** - single-user demo application.
- **No multi-tenancy** - all knowledge bases are accessible without authentication.

## Resume Highlights

- **End-to-end RAG pipeline**: upload → parse → chunk → embed → index → retrieve, fully functional
- **Hybrid search**: weighted merge of vector similarity and keyword matching with score normalization and deduplication
- **Pluggable architecture**: embedding providers, rerankers, storage backends, and parsers are all pluggable via abstract base classes
- **Multiple document formats**: PDF (PyMuPDF), DOCX, Markdown, TXT with encoding fallback
- **Clean separation of concerns**: route → manager → knowledge base → providers, no business logic in route handlers
- **Graceful degradation**: Milvus unavailable → clear error message, not crash
- **Unified error format**: all API errors return structured `{code, message, details}` format
- **File status state machine**: 8 states with failure tracking (reason + stage + timestamp)
- **Zero-dependency testing**: FakeEmbeddingProvider + Milvus Lite + SQLite, no Docker or API keys needed

## Project Structure

```
yuxi-rag-mini/
├── backend/
│   └── app/
│       ├── api/                    # FastAPI route handlers
│       │   ├── health_routes.py
│       │   ├── kb_routes.py
│       │   ├── file_routes.py
│       │   └── query_routes.py
│       ├── core/
│       │   ├── config.py           # Pydantic Settings
│       │   ├── errors.py
│       │   └── logging.py
│       └── rag/
│           ├── base.py             # KnowledgeBase ABC + FileStatus
│           ├── factory.py          # KnowledgeBaseFactory
│           ├── manager.py          # KnowledgeBaseManager
│           ├── schemas.py          # Pydantic request/response models
│           ├── backends/
│           │   └── milvus_kb.py    # MilvusKB implementation
│           ├── parser/
│           │   ├── base.py         # ParseResult + BaseParser
│           │   ├── factory.py      # parse_file dispatcher
│           │   ├── pdf_parser.py   # PyMuPDF (primary) + pypdf (fallback)
│           │   ├── docx_parser.py
│           │   ├── markdown_parser.py
│           │   └── text_parser.py
│           ├── chunking/
│           │   ├── text_chunker.py
│           │   └── markdown_chunker.py
│           ├── providers/
│           │   ├── embedding/      # OpenAI/Ollama/HuggingFace/Fake
│           │   └── rerank/         # DummyReranker + ApiReranker
│           ├── graphrag/
│           │   └── interfaces.py   # Placeholder ABCs
│           ├── repositories/
│           │   ├── kb_repository.py
│           │   ├── file_repository.py
│           │   └── chunk_repository.py
│           └── storage/
│               ├── database.py
│               ├── models.py
│               ├── local_file_storage.py
│               └── minio_storage.py
├── frontend/                       # Vue 3 + Vite + TypeScript
├── tests/
│   ├── test_parser.py
│   ├── test_chunker.py
│   ├── test_hybrid_search.py
│   └── test_pipeline.py
├── docker-compose.yml              # Milvus stack
├── pyproject.toml
└── .env.example
```

## License

MIT
