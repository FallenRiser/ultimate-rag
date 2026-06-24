# Ultimate RAG

Enterprise-grade, **agentic**, multi-retrieval RAG platform.

- **Agentic** query analysis, rewrite/decompose, multi-retrieval, grading loop (LangGraph)
- **Retrieval modes:** semantic · BM25 · graph · hybrid · hybrid+graph
- **Pluggable everything** — vector DB, graph DB, LLM, embeddings, parser, chunker, reranker, cache all behind a base class + config
- **Providers today:** pgvector + Qdrant + Chroma · Azure OpenAI + OpenAI + Ollama embeddings · OpenAI/vLLM/Ollama LLMs · Docling + custom parsing · BGE reranker · Apache Ignite cache
- **Relational:** Postgres or SQLite (config-selected)
- **Enterprise:** per-user document isolation (`X-User-Id` header), document versioning, metadata filtering, citations w/ confidence, chat sessions w/ memory
- **Ops:** in-process ingestion (background task or `POST /ingest/sync`), MLflow tracing, Rich logging, Pydantic everywhere

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design and phased roadmap.

> **Status:** Phase 0 — skeleton. Folder structure, base classes, config schema, and
> Pydantic models are in place. Concrete implementations are stubs (`NotImplementedError`).

## Layout
```
app/
  api/v1/routes/   HTTP routes (thin)
  services/        business logic (embeddings, llm, parsing, chunking, reranking,
                   retrieval, ingestion, agent, citation)
  repositories/    DB connections (vector, graph, cache, relational)
  models/          Pydantic models
  observability/   logging (Rich) + tracing (MLflow)
  utils/           config loader, registry, helpers
config/config.yaml  runtime configuration
logs/               Rich file logs
```

## Quickstart (dev)
```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -e ".[parsing,dev]"
cp .env.example .env                                    # set OPENAI_API_KEY (+ OPENAI_BASE_URL)
uvicorn app.main:app --reload
```

Ingestion runs in-process — no separate worker or Redis. The async `POST /ingest`
returns immediately and processes in a FastAPI background task; `POST /ingest/sync`
runs the pipeline inline and returns the final status.

**Zero-infra setup:** set `database.provider: sqlite` and `vector_store.provider: chroma`
in `config/config.yaml` to run with no external services (semantic retrieval only).

Configuration is driven by `config/config.yaml`. The only env secrets are
`OPENAI_API_KEY` / `OPENAI_BASE_URL`, used only when an `openai` provider is selected.
There is no auth layer: callers pass an `X-User-Id` header and every repository
filters on it for tenant isolation (absent → `default`).
