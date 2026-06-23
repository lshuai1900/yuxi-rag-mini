# Yuxi RAG Extracted

A production-grade **RAG knowledge base system**, extracted and refactored from the [Yuxi](https://github.com/xerrors/Yuxi) project's RAG subsystem. Implements Milvus-native BM25 hybrid search, ragflow-like chunking, unified document parsing, and double-write consistency.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Console (Vue 3)                      │
│   KB Management / File Upload / Status Tracking / Query     │
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
│  ┌──▼──────────┐  ┌─▼────────────┐  ┌──▼──────────────┐   │
│  │ Unified      │  │ragflow_like  │  │ Embedding        │   │
│  │ Parser       │  │ Chunking     │  │ Provider         │   │
│  │(PDF/DOCX/MD/ │  │ dispatcher   │  │(OpenAI/Ollama/   │   │
│  │ TXT→Markdown)│  │ nlp/presets  │  │ HF/Fake)          │   │
│  └──────────────┘  └──────────────┘  └──────┬───────────┘   │
│                                              │                │
│  ┌──────────┐  ┌──────────┐                 │                │
│  │ Reranker │  │ GraphRAG │                 │                │
│  │ (Pluggable)│  │Reserved │                 │                │
│  └──────────┘  └──────────┘                 │                │
└────────────────────────────────────────────┼────────────────┘
                    ┌─────────────────────────┼─────────────────┐
            ┌───────▼──────┐        ┌────────▼────────┐  ┌──────▼──────┐
            │ Milvus (per- │        │ SQLite /         │  │ Local /     │
            │ KB collection│        │ PostgreSQL       │  │ MinIO       │
            │ + BM25 sparse│        │ + chunk repo     │  │ Storage     │
            └──────────────┘        └──────────────────┘  └─────────────┘
```

## Core Features

### Milvus Native Hybrid Search

Uses Milvus's built-in `FunctionType.BM25` with `SPARSE_FLOAT_VECTOR` and `SPARSE_INVERTED_INDEX` for native BM25 keyword search. Combined dense vector search via `AnnSearchRequest` + `WeightedRanker` for server-side fusion.

```
vector_request = AnnSearchRequest(embedding, "embedding", COSINE)
bm25_request   = AnnSearchRequest(query_text, "content_sparse", BM25)
results = collection.hybrid_search(
    reqs=[vector_request, bm25_request],
    rerank=WeightedRanker(vector_weight=0.7, bm25_weight=0.3),
)
```

### Retrieval Configuration (`MilvusRetrievalConfig`)

18-field dataclass controlling search behavior:

| Field | Default | Description |
|-------|---------|-------------|
| `search_mode` | `hybrid` | vector / keyword / hybrid |
| `final_top_k` | 10 | Final result count |
| `similarity_threshold` | 0.0 | Minimum score filter |
| `bm25_top_k` | 20 | BM25 recall candidates |
| `vector_weight` | 0.7 | Dense vector weight in fusion |
| `bm25_weight` | 0.3 | BM25 weight in fusion |
| `bm25_drop_ratio_search` | 0.2 | BM25 drop ratio for low-quality terms |
| `recall_top_k` | 20 | Pre-fusion candidate count |
| `use_reranker` | false | Enable reranking step |

Plus graph retrieval fields reserved: `use_graph_retrieval`, `graph_entity_top_k`, `graph_triple_top_k`, `graph_max_nodes`, `graph_top_k`, `graph_weight`, `ppr_damping`.

### Collection-per-KB Design

Each knowledge base gets its own Milvus collection with:

- **Schema**: id (PK), content (Chinese-analyzed VARCHAR), chunk_id, file_id, chunk_index, embedding (FLOAT_VECTOR), content_sparse (SPARSE_FLOAT_VECTOR via BM25 function)
- **Dense index**: IVF_FLAT / COSINE
- **Sparse index**: SPARSE_INVERTED_INDEX / BM25 (DAAT_MAXSCORE)

### File Lifecycle

```
uploaded → parsing → parsed → indexing → indexed
                     ↘ error_parsing    ↘ error_indexing
```

Two-phase pipeline:
1. **parse_file**: source file → markdown (via unified parser) → save `markdown_file` to storage
2. **index_file**: read `markdown_file` → chunk (ragflow_like) → embed → **double-write** to Milvus + PG

### Double-Write Consistency

Milvus insert and PostgreSQL upsert run concurrently. On failure, both stores are rolled back:

```python
pg_task = asyncio.create_task(chunk_repo.batch_upsert(records))
milvus_task = asyncio.create_task(collection.insert(entities))
results = await asyncio.gather(pg_task, milvus_task, return_exceptions=True)
if errors:
    # Rollback: delete from PG + Milvus
    await chunk_repo.delete_by_file_id(file_id)
    await collection.delete(f'file_id == "{file_id}"')
```

### Query Offload

All blocking Milvus queries are offloaded to thread pools with semaphore-based concurrency control (default: 8 concurrent):

```python
async def _run_milvus_query_io(func, *args, **kwargs):
    semaphore = _get_milvus_query_offload_semaphore()  # Semaphore(8)
    await semaphore.acquire()
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    finally:
        semaphore.release()
```

### ragflow-like Chunking System

Ported from Yuxi's `chunking/ragflow_like/` module:

```
dispatcher.py → chunk_markdown(markdown, file_id, filename, params) → list[dict]
    ├── presets.py → resolve_chunk_processing_params(preset_id)
    │   Presets: general (512), qa (512), book (1024), laws (2048), semantic (512), separator (512)
    └── parsers/general.py → chunk_markdown(md_content) → list[str]
        └── nlp.py → naive_merge(sections, token_limit, delimiter, overlap)
                  ├── hard_split_by_token_limit(text, limit, hard_limit)
                  ├── bullets_category(sections) → depth
                  ├── tree_merge / hierarchical_merge
                  └── count_tokens(text) → int
```

Chunk records include position metadata: `start_char_pos`, `end_char_pos`, `start_token_pos`, `end_token_pos`.

### Unified Document Parser

Single entry point that converts any supported file format to Markdown:

```python
# Parse file → markdown string
markdown = await Parser.aparse(file_path)

# Parse file → structured result
result = await parse_source_to_markdown(file_path, params={"filename": "doc.pdf"})
# Returns MarkdownParseResult(markdown="...", file_ext=".pdf", artifacts={...})
```

Supported formats: PDF (PyMuPDF), DOCX (python-docx), Markdown, TXT. Unsupported types fall back to plain text reading.

Optional adapter pattern for heavy external processors (MinerU, PaddleX, OCR) — loaded lazily, doesn't break startup if unavailable.

### GraphRAG Fields Reserved

`KnowledgeChunkModel` includes graph fields (not implemented):
- `graph_indexed` (Boolean)
- `ent_ids` (JSON list of entity IDs)
- `tags` (JSON list of tags)
- `extraction_result` (JSON graph extraction output)

Repository methods: `mark_graph_indexed()`, `reset_graph_state_by_kb_id()`, `list_unindexed_for_graph()`.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Pydantic v2, SQLAlchemy (async) |
| Frontend | Vue 3, Vite, TypeScript (RAG Console dark theme) |
| Vector DB | Milvus Lite / Standalone (BM25 + Dense Vector) |
| Metadata DB | SQLite (aiosqlite) / PostgreSQL (asyncpg) |
| File Storage | Local filesystem / MinIO |
| Embedding | OpenAI-compatible, Ollama, HuggingFace, Fake (testing) |
| Document Parsing | PyMuPDF, python-docx, custom MD/TXT parsers |

## Quick Start

### Option A: Zero Dependencies (Milvus Lite + SQLite)

```bash
cd backend
pip install -e ..
cp ../.env.example .env
uvicorn app.main:app --reload --port 8000

cd ../frontend && npm install && npm run dev
```

### Option B: Full Stack (Docker + Milvus Standalone)

```bash
docker compose up -d milvus
cd backend && pip install -e .. && uvicorn app.main:app --port 8000
cd ../frontend && npm install && npm run dev
```

## API Examples

```bash
# Create KB
curl -X POST http://localhost:8000/api/kb \
  -H "Content-Type: application/json" \
  -d '{"name": "My KB", "description": ""}'

# Upload file
curl -X POST http://localhost:8000/api/kb/{kb_id}/files/upload -F "file=@doc.pdf"

# Parse (file → markdown)
curl -X POST http://localhost:8000/api/kb/{kb_id}/files/{file_id}/parse

# Index (markdown → chunk → embed → Milvus+PG)
curl -X POST http://localhost:8000/api/kb/{kb_id}/files/{file_id}/index

# One-step ingest (parse + index)
curl -X POST http://localhost:8000/api/kb/{kb_id}/files/{file_id}/ingest

# Query with retrieval config
curl -X POST http://localhost:8000/api/kb/{kb_id}/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "retrieval_config": {
      "search_mode": "hybrid",
      "final_top_k": 10,
      "similarity_threshold": 0.0,
      "vector_weight": 0.7,
      "bm25_weight": 0.3,
      "bm25_top_k": 20,
      "recall_top_k": 20
    }
  }'
```

## Configuration

Key settings (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `fake` | openai_compatible / ollama / huggingface / fake |
| `EMBEDDING_MODEL` | bge-m3 | Model name |
| `EMBEDDING_DIMENSION` | 128 | Vector dimension |
| `MILVUS_URI` | data/milvus.db | Connection URI |
| `DEFAULT_SEARCH_MODE` | hybrid | Default query mode |
| `DEFAULT_VECTOR_WEIGHT` | 0.7 | Hybrid fusion vector weight |
| `DEFAULT_BM25_WEIGHT` | 0.3 | Hybrid fusion BM25 weight |
| `MILVUS_QUERY_OFFLOAD_SEMAPHORE` | 8 | Max concurrent Milvus queries |

> **WARNING**: `fake` embedding generates deterministic but meaningless vectors. Use real providers for production.

## Testing

```bash
cd backend && python -m pytest ../tests/ -v
```

57 tests covering: retrieval config, ragflow chunking, unified parser, file status, import checks, Milvus schema design, chunk model, parser, chunker.

## Project Structure

```
yuxi-rag-mini/
├── backend/app/
│   ├── api/                        # Route handlers
│   ├── core/config.py              # Settings (18+ config vars)
│   └── rag/
│       ├── base.py                 # KnowledgeBase ABC + FileStatus (7 states)
│       ├── schemas.py              # MilvusRetrievalConfig (18 fields) + API models
│       ├── backends/milvus_kb.py   # MilvusKB: BM25, WeightedRanker, double-write
│       ├── parser/
│       │   ├── unified.py           # Parser.aparse() → markdown string
│       │   ├── adapter.py           # Optional processor adapters (lazy load)
│       │   └── factory.py           # parse_file dispatcher
│       ├── chunking/
│       │   └── ragflow_like/        # Ported from Yuxi
│       │       ├── dispatcher.py    # Main entry: chunk_markdown()
│       │       ├── nlp.py           # naive_merge, hard_split, count_tokens
│       │       ├── presets.py       # 6 preset configs
│       │       └── parsers/general.py  # Markdown section splitter
│       ├── repositories/
│       │   └── chunk_repository.py  # batch_upsert, graph methods
│       ├── storage/models.py        # KnowledgeChunkModel (+ graph fields)
│       └── providers/               # Embedding + reranker pluggable
├── frontend/src/                    # Vue 3 RAG Console (dark theme)
├── tests/                           # 9 test files, 57 tests
├── pyproject.toml
└── .env.example
```

## Alignment with Yuxi RAG Subsystem

This project is extracted from the Yuxi project's RAG subsystem with these alignment points:

| Component | Yuxi Source | Alignment Status |
|-----------|------------|------------------|
| MilvusKB core | `implementations/milvus.py` | BM25 schema, hybrid search, query offload, double-write |
| Retrieval Config | `schemas.py` | 18-field MilvusRetrievalConfig |
| File Lifecycle | `base.py` | 7-state machine (error_parsing/error_indexing) |
| Chunking | `chunking/ragflow_like/` | dispatcher, nlp, presets, general parser |
| Unified Parser | `parser/unified.py` | aparse → markdown, optional adapters |
| Repository | `repositories/chunk_repository.py` | batch_upsert, graph fields |
| GraphRAG fields | model + repository | Reserved (not implemented) |

## Limitations

- **No user/auth system** — single-user console
- **GraphRAG** — interface reserved, not implemented
- **Reranker** — DummyReranker by default; real reranker can be plugged in
- **OCR** — adapter pattern exists but no OCR provider configured by default
- **Multi-tenancy** — not implemented

## License

MIT
