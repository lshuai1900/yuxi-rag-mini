# Yuxi RAG Extracted

从 [Yuxi](https://github.com/xerrors/Yuxi) 项目 RAG 子系统提取并重构的**生产级 RAG 知识库**。实现了 Milvus 原生 BM25 混合检索、ragflow-like 分块、统一文档解析和双写一致性。

## 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                   RAG 控制台 (Vue 3)                         │
│      知识库管理 / 文件上传 / 状态追踪 / 检索查询              │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP API
┌────────────────────────────▼────────────────────────────────┐
│                      FastAPI 后端                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ KB 路由   │  │文件路由   │  │查询路由   │  │健康检查路由   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────────┘ │
│       └──────────────┼─────────────┘                          │
│              ┌───────▼────────┐                               │
│              │ KnowledgeBase  │                               │
│              │    Manager     │                               │
│              └───────┬────────┘                               │
│     ┌────────────────┼────────────────┐                      │
│  ┌──▼──────────┐  ┌─▼────────────┐  ┌──▼──────────────┐   │
│  │ 统一解析器   │  │ragflow_like  │  │ Embedding        │   │
│  │ Parser      │  │ 分块系统      │  │ Provider         │   │
│  │(PDF/DOCX/MD │  │ dispatcher   │  │(OpenAI/Ollama/   │   │
│  │ /TXT→MD)    │  │ nlp/presets  │  │ HF/Fake)          │   │
│  └──────────────┘  └──────────────┘  └──────┬───────────┘   │
│                                              │                │
│  ┌──────────┐  ┌──────────┐                 │                │
│  │ Reranker │  │ GraphRAG │                 │                │
│  │ (可插拔)  │  │ (预留)   │                 │                │
│  └──────────┘  └──────────┘                 │                │
└────────────────────────────────────────────┼────────────────┘
                    ┌─────────────────────────┼─────────────────┐
            ┌───────▼──────┐        ┌────────▼────────┐  ┌──────▼──────┐
            │ Milvus (每KB │        │ SQLite /        │  │ 本地 /      │
            │ 独立Collection│        │ PostgreSQL      │  │ MinIO 存储  │
            │ + BM25稀疏向量│        │ + chunk仓库     │  │             │
            └──────────────┘        └──────────────────┘  └─────────────┘
```

## 核心特性

### Milvus 原生混合检索

使用 Milvus 内置 `FunctionType.BM25` 配合 `SPARSE_FLOAT_VECTOR` 和 `SPARSE_INVERTED_INDEX` 实现原生 BM25 关键词检索。通过 `AnnSearchRequest` + `WeightedRanker` 实现服务端稠密/稀疏向量融合：

```python
vector_request = AnnSearchRequest(embedding, "embedding", COSINE)
bm25_request   = AnnSearchRequest(query_text, "content_sparse", BM25)
results = collection.hybrid_search(
    reqs=[vector_request, bm25_request],
    rerank=WeightedRanker(vector_weight=0.7, bm25_weight=0.3),
)
```

### 检索配置 (`MilvusRetrievalConfig`)

18 字段数据类，精细控制检索行为：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `search_mode` | `hybrid` | vector / keyword / hybrid |
| `final_top_k` | 10 | 最终返回结果数 |
| `similarity_threshold` | 0.0 | 最低相似度过滤 |
| `bm25_top_k` | 20 | BM25 召回候选数 |
| `vector_weight` | 0.7 | 稠密向量融合权重 |
| `bm25_weight` | 0.3 | BM25 融合权重 |
| `bm25_drop_ratio_search` | 0.2 | BM25 低质量词项丢弃比例 |
| `recall_top_k` | 20 | 融合前候选数 |
| `use_reranker` | false | 启用重排序 |

图检索预留字段：`use_graph_retrieval`, `graph_entity_top_k`, `graph_triple_top_k`, `graph_max_nodes`, `graph_top_k`, `graph_weight`, `ppr_damping`。

### 每个知识库独立 Collection

每个知识库拥有独立的 Milvus Collection，Schema 包含：

- **字段**：id (主键), content (中文分词 VARCHAR), chunk_id, file_id, chunk_index, embedding (FLOAT_VECTOR), content_sparse (SPARSE_FLOAT_VECTOR，由 BM25 函数生成)
- **稠密索引**：IVF_FLAT / COSINE
- **稀疏索引**：SPARSE_INVERTED_INDEX / BM25 (DAAT_MAXSCORE)

### 文件生命周期

```
uploaded → parsing → parsed → indexing → indexed
                     ↘ error_parsing    ↘ error_indexing
```

两阶段流水线：
1. **parse_file**：源文件 → Markdown（统一解析器）→ 保存 `markdown_file` 到存储
2. **index_file**：读取 `markdown_file` → 分块（ragflow_like）→ 向量化 → **双写** Milvus + PG

### 双写一致性

Milvus 写入和 PostgreSQL upsert 并发执行，任一失败则双向回滚：

```python
pg_task = asyncio.create_task(chunk_repo.batch_upsert(records))
milvus_task = asyncio.create_task(collection.insert(entities))
results = await asyncio.gather(pg_task, milvus_task, return_exceptions=True)
if errors:
    # 回滚：删除 PG + Milvus 中的数据
    await chunk_repo.delete_by_file_id(file_id)
    await collection.delete(f'file_id == "{file_id}"')
```

### 查询卸载

所有阻塞式 Milvus 查询通过线程池卸载，信号量控制并发（默认 8 路）：

```python
async def _run_milvus_query_io(func, *args, **kwargs):
    semaphore = _get_milvus_query_offload_semaphore()  # Semaphore(8)
    await semaphore.acquire()
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    finally:
        semaphore.release()
```

### ragflow-like 分块系统

从 Yuxi 的 `chunking/ragflow_like/` 模块移植：

```
dispatcher.py → chunk_markdown(markdown, file_id, filename, params) → list[dict]
    ├── presets.py → resolve_chunk_processing_params(preset_id)
    │   预设：general (512), qa (512), book (1024), laws (2048), semantic (512), separator (512)
    └── parsers/general.py → chunk_markdown(md_content) → list[str]
        └── nlp.py → naive_merge(sections, token_limit, delimiter, overlap)
                  ├── hard_split_by_token_limit(text, limit, hard_limit)
                  ├── bullets_category(sections) → depth
                  ├── tree_merge / hierarchical_merge
                  └── count_tokens(text) → int
```

分块记录包含位置元数据：`start_char_pos`, `end_char_pos`, `start_token_pos`, `end_token_pos`。

### 统一文档解析器

单一入口，将任意支持格式转换为 Markdown：

```python
# 解析文件 → Markdown 字符串
markdown = await Parser.aparse(file_path)

# 解析文件 → 结构化结果
result = await parse_source_to_markdown(file_path, params={"filename": "doc.pdf"})
# 返回 MarkdownParseResult(markdown="...", file_ext=".pdf", artifacts={...})
```

支持格式：PDF (PyMuPDF)、DOCX (python-docx)、Markdown、TXT。不支持的格式回退为纯文本读取。

可选适配器模式支持重型外部处理器（MinerU、PaddleX、OCR）——懒加载，不可用时不影响启动。

### GraphRAG 字段预留

`KnowledgeChunkModel` 包含图字段（未实现）：
- `graph_indexed` (Boolean) — 是否已建图索引
- `ent_ids` (JSON) — 实体 ID 列表
- `tags` (JSON) — 标签列表
- `extraction_result` (JSON) — 图抽取结果

仓库方法：`mark_graph_indexed()`, `reset_graph_state_by_kb_id()`, `list_unindexed_for_graph()`。

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | FastAPI, Pydantic v2, SQLAlchemy (async) |
| 前端 | Vue 3, Vite, TypeScript（RAG 控制台暗色主题） |
| 向量数据库 | Milvus Lite / Standalone（BM25 + 稠密向量） |
| 元数据数据库 | SQLite (aiosqlite) / PostgreSQL (asyncpg) |
| 文件存储 | 本地文件系统 / MinIO |
| 向量化 | OpenAI 兼容接口, Ollama, HuggingFace, Fake（测试用） |
| 文档解析 | PyMuPDF, python-docx, 自定义 MD/TXT 解析器 |

## 快速开始

### 方式 A：零依赖启动（Milvus Lite + SQLite）

```bash
cd backend
pip install -e ..
cp ../.env.example .env
uvicorn app.main:app --reload --port 8000

cd ../frontend && npm install && npm run dev
```

### 方式 B：完整部署（Docker + Milvus Standalone）

```bash
docker compose up -d milvus
cd backend && pip install -e .. && uvicorn app.main:app --port 8000
cd ../frontend && npm install && npm run dev
```

## API 示例

```bash
# 创建知识库
curl -X POST http://localhost:8000/api/kb \
  -H "Content-Type: application/json" \
  -d '{"name": "我的知识库", "description": ""}'

# 上传文件
curl -X POST http://localhost:8000/api/kb/{kb_id}/files/upload -F "file=@doc.pdf"

# 解析（文件 → Markdown）
curl -X POST http://localhost:8000/api/kb/{kb_id}/files/{file_id}/parse

# 索引（Markdown → 分块 → 向量化 → 写入 Milvus+PG）
curl -X POST http://localhost:8000/api/kb/{kb_id}/files/{file_id}/index

# 一键入库（解析 + 索引）
curl -X POST http://localhost:8000/api/kb/{kb_id}/files/{file_id}/ingest

# 带检索配置的查询
curl -X POST http://localhost:8000/api/kb/{kb_id}/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习？",
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

## 配置项

关键环境变量（详见 `.env.example`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_PROVIDER` | `fake` | openai_compatible / ollama / huggingface / fake |
| `EMBEDDING_MODEL` | bge-m3 | 向量化模型名称 |
| `EMBEDDING_DIMENSION` | 128 | 向量维度 |
| `MILVUS_URI` | data/milvus.db | Milvus 连接地址 |
| `DEFAULT_SEARCH_MODE` | hybrid | 默认检索模式 |
| `DEFAULT_VECTOR_WEIGHT` | 0.7 | 混合检索向量权重 |
| `DEFAULT_BM25_WEIGHT` | 0.3 | 混合检索 BM25 权重 |
| `MILVUS_QUERY_OFFLOAD_SEMAPHORE` | 8 | Milvus 最大并发查询数 |

> **注意**：`fake` 向量化生成确定性但无意义的向量，仅用于测试流水线，不可用于真实 RAG 场景。

## 测试

```bash
cd backend && python -m pytest ../tests/ -v
```

57 个测试覆盖：检索配置、ragflow 分块、统一解析器、文件状态、模块导入检查、Milvus Schema 设计、chunk 模型、解析器、分块器。

## 项目结构

```
yuxi-rag-mini/
├── backend/app/
│   ├── api/                          # 路由处理器
│   ├── core/config.py                # 配置（18+ 配置项）
│   └── rag/
│       ├── base.py                   # KnowledgeBase ABC + FileStatus（7 状态）
│       ├── schemas.py                # MilvusRetrievalConfig（18 字段）+ API 模型
│       ├── backends/milvus_kb.py     # MilvusKB：BM25、WeightedRanker、双写
│       ├── parser/
│       │   ├── unified.py            # Parser.aparse() → Markdown 字符串
│       │   ├── adapter.py            # 可选处理器适配器（懒加载）
│       │   └── factory.py            # parse_file 分发器
│       ├── chunking/
│       │   └── ragflow_like/         # 从 Yuxi 移植
│       │       ├── dispatcher.py     # 主入口：chunk_markdown()
│       │       ├── nlp.py            # naive_merge, hard_split, count_tokens
│       │       ├── presets.py        # 6 种分块预设
│       │       └── parsers/general.py  # Markdown 章节分割器
│       ├── repositories/
│       │   └── chunk_repository.py   # batch_upsert、图相关方法
│       ├── storage/models.py         # KnowledgeChunkModel（+ 图字段）
│       └── providers/                # 向量化 + 重排序（可插拔）
├── frontend/src/                     # Vue 3 RAG 控制台（暗色主题）
├── tests/                            # 9 个测试文件，57 个测试
├── pyproject.toml
└── .env.example
```

## 与 Yuxi RAG 子系统对齐情况

| 组件 | Yuxi 源码 | 对齐状态 |
|------|-----------|----------|
| MilvusKB 核心 | `implementations/milvus.py` | BM25 Schema、混合检索、查询卸载、双写 |
| 检索配置 | `schemas.py` | 18 字段 MilvusRetrievalConfig |
| 文件生命周期 | `base.py` | 7 状态机（error_parsing/error_indexing） |
| 分块系统 | `chunking/ragflow_like/` | dispatcher、nlp、presets、general 解析器 |
| 统一解析器 | `parser/unified.py` | aparse → Markdown、可选适配器 |
| 仓库层 | `repositories/chunk_repository.py` | batch_upsert、图字段 |
| GraphRAG 字段 | model + repository | 预留（未实现） |

## 已知限制

- **无用户/认证系统** — 单用户控制台
- **GraphRAG** — 接口预留，未实现
- **Reranker** — 默认 DummyReranker；可插拔真实重排序模型
- **OCR** — 适配器模式已就绪，默认未配置 OCR 提供者
- **多租户** — 未实现

## 许可证

MIT
