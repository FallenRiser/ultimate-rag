# Ultimate RAG

Enterprise-grade, **agentic**, multi-retrieval RAG platform with a real **GraphRAG**
knowledge graph. Every external dependency sits behind a small abstract base class
selected in `config/config.yaml`, so swapping a vector DB / graph store / LLM / embedder /
parser / reranker / cache is a one-file change.

- **Agentic** query analysis, rewrite/decompose, multi-retrieval, relevance grading loop (LangGraph)
- **Retrieval modes:** `semantic` · `bm25` · `graph` · `hybrid` · `hybrid_graph` · `graph_global`
- **GraphRAG (local + global):** entity + relation extraction with descriptions, **cross-document
  entity merging**, **semantic entity matching** (embedded entities, not string match),
  **multi-hop traversal**, **community detection + LLM reports** (thematic/global queries), and
  **dual-level** retrieval (entity chunks blended with community context)
- **Pluggable everything** — vector store, graph store, LLM, embeddings, parser, chunker, reranker,
  cache, relational DB, all behind a base class + a one-function factory keyed on config
- **Providers today:**
  - Vector: **pgvector** (semantic) · **Chroma** (semantic) · **Qdrant** (semantic + BM25)
  - Graph: **NetworkX** (local JSON files, zero-infra default) · **Apache AGE** (Postgres)
  - Embeddings: Azure OpenAI · OpenAI · Ollama   ·   LLM: OpenAI · vLLM · Ollama (all OpenAI-compatible)
  - Parsing: Docling-serve · custom (pypdf/python-docx/python-pptx/openpyxl/OCR)
  - Reranker: BGE (Qwen3/Cohere optional)   ·   Cache: Apache Ignite
- **Relational:** Postgres or SQLite (config-selected)
- **Capability-aware:** the active vector store decides which modes exist (BM25/hybrid rejected on
  semantic-only stores — no FTS fake-out); graph modes require a graph store
- **Enterprise:** per-user (tenant) isolation enforced at the repository layer, document versioning,
  LLM metadata extraction + auto-filtering, citations with confidence, chat sessions with memory
- **Ops:** in-process ingestion (background task or `POST /ingest/sync`), MLflow tracing, Rich logging,
  Pydantic at every boundary

**Docs:** **[CONFIG_AND_API.md](CONFIG_AND_API.md)** — when to use which config + what every endpoint does ·
**[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** — layer-by-layer design, file map, how to add a provider, coding
standards · [ARCHITECTURE.md](ARCHITECTURE.md) — original design decisions and roadmap.

## Layout
```
app/
  api/v1/routes/   thin HTTP routes (health, documents, ingestion, query, chat)
  services/        business logic: embeddings, llm, parsing, chunking, reranking,
                   retrieval, ingestion, graph (GraphRAG build), agent, citation
  repositories/    infrastructure access: vector, graph, cache, relational, storage
  models/          Pydantic models (validation/serialization at boundaries)
  prompts/         all LLM prompts (ingestion, retrieval, agent)
  observability/   logging (Rich) + tracing (MLflow) + timing
  utils/           config loader, hashing, text helpers
config/config.yaml  single source of runtime configuration
logs/               Rich rotating file logs
```

## Quickstart (dev)
```bash
python -m venv .venv && .venv\Scripts\activate          # bash: source .venv/Scripts/activate
pip install -e ".[parsing,dev]"
cp .env.example .env                                    # only needed for openai providers
uvicorn app.main:app --reload                           # http://localhost:8000/docs
```

**Zero-infra stack (no Docker, no external services):** set these in `config/config.yaml` and use
local Ollama for the LLM + embeddings —

```yaml
database:      { provider: sqlite }
vector_store:  { provider: chroma }      # semantic only (no BM25)
graph_store:   { provider: networkx }    # local JSON files
```

This runs the full pipeline — including GraphRAG — entirely on local files. Use `qdrant` if you want
BM25/hybrid, and `postgres` + `age` if you want everything in one ACID database.

## API (prefix `/api/v1`)
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness |
| `POST` | `/ingest` | ingest (background task), returns immediately |
| `POST` | `/ingest/sync` | ingest inline, returns final status |
| `POST` | `/ingest/upload` · `/ingest/by-filename` · `/ingest/bytes` | file/bytes ingestion variants |
| `GET` | `/ingest/status/{document_id}` | ingestion status |
| `GET` | `/documents` · `/documents/{id}` · `/documents/{id}/chunks` | list / get / chunks |
| `DELETE` | `/documents/{id}` | delete a document (cascades chunks + graph) |
| `POST` | `/documents/graph/rebuild-communities` | rebuild GraphRAG community reports (global/thematic mode) |
| `POST` | `/query` | one-shot RAG query (optionally agentic) |
| `POST` | `/chat` | chat turn with session memory |
| `GET` | `/chat/sessions` · `/chat/sessions/{id}/messages` | paginated sessions / messages |
| `DELETE` | `/chat/sessions/{id}` | delete a session (cascades messages) |

## Tenancy & secrets
There is **no auth layer**. Callers pass `user_id` **in the request payload** (JSON body field,
multipart form field, or query param); every repository read/write filters on it for tenant
isolation. Absent → `"default"`. Put a gateway/IdP in front for production.

Configuration lives in `config/config.yaml`. The only env secrets are `OPENAI_API_KEY` /
`OPENAI_BASE_URL` (used only when an `openai`/`azure_openai` provider is selected) and the Postgres
DSN; env overrides YAML (prefix `RAG_`, nested keys via `__`, e.g. `RAG_LLM__API_KEY`).
