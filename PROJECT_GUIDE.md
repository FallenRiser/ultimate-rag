# Ultimate RAG — Project Guide

A practical map of the codebase: what every part does, how the layers fit together, **how to add
new things the right way**, and the standards all code must follow. Read this before making changes.

> The terse rulebook is [CLAUDE.md](CLAUDE.md); the original design decisions are in
> [ARCHITECTURE.md](ARCHITECTURE.md). This guide is the day-to-day developer handbook.

---

## 1. What this is

An agentic, multi-retrieval RAG platform. You ingest documents; they're parsed → chunked → embedded →
indexed into a vector store, and (when a graph store is enabled) mined into a **knowledge graph**.
At query time the system retrieves with one of several strategies, optionally orchestrated by a
LangGraph agent, then synthesizes a cited answer.

Two things define the design:

1. **Layered, one job per layer.** Routes do HTTP, services do logic, repositories do I/O, models
   validate boundaries. Logic never imports a vendor SDK directly.
2. **Capability-aware, config-selected providers.** Every swappable dependency hides behind an
   abstract base class chosen by a string in `config/config.yaml`. The *active* provider decides what's
   possible (e.g. a semantic-only vector store rejects BM25 instead of faking it).

### Request lifecycle — query
```
POST /api/v1/query
  → routes/query.py            (thin: parse QueryRequest, call the pipeline/agent)
  → services/retrieval/pipeline.py   embed → auto-filter → retrieve → rerank → synthesize → cite
        or services/agent/graph.py   (agentic: analyze → rewrite/decompose → retrieve → grade → loop)
  → services/retrieval/service.py    mode dispatch (semantic/bm25/graph/hybrid/hybrid_graph/graph_global)
  → repositories/vector/* + repositories/graph/*   (user_id-filtered I/O)
  → services/citation/builder.py     answer + source_chunks + confidence
```

### Request lifecycle — ingest
```
POST /api/v1/ingest[/sync]
  → routes/ingestion.py        (thin)
  → services/ingestion/service.py    register (dedup) → process:
        parse → chunk → metadata-extract → embed → vector insert
        → GraphRAG: extract entities+relations → embed entities → upsert graph
          → (optional) summarize descriptions, rebuild communities
  → repositories/vector/* + repositories/graph/* + repositories/relational/*
```

---

## 2. The five layers

| Layer | Directory | Job | May import |
|---|---|---|---|
| **API** | `app/api/v1/routes/` | HTTP only: parse request → call a service → return a model. **No business logic.** | services, models |
| **Services** | `app/services/` | All business logic, grouped by concern. Talks to **base classes**, never vendor SDKs. | other services, repositories (via base/factory), models, prompts, utils |
| **Repositories** | `app/repositories/` | Infrastructure access (vector DBs, graph stores, relational DB, cache, file storage). | models, utils, vendor SDKs |
| **Models** | `app/models/` | Pydantic models — validation/serialization at every boundary (HTTP, LLM output). | nothing app-specific |
| **Cross-cutting** | `app/observability/`, `app/utils/`, `app/prompts/`, `config/` | logging, tracing, timing, config, hashing, text helpers, LLM prompts. | — |

**Hard rule:** business logic depends only on **interfaces** (`BaseVectorDB`, `BaseLLMProvider`, …),
never concrete vendor classes. The factory resolves the concrete class from config.

---

## 3. The provider pattern (the main extension mechanism)

Every swappable concern follows the same three-part shape:

```
base.py        an ABC with async methods + explicit type hints   (the contract)
<impl>.py      one file per concrete implementation               (qdrant.py, ollama.py, …)
factory.py     ONE function mapping config `provider` string → class instance
```

- **ABC location:** infrastructure ABCs live in `app/repositories/base.py`
  (`BaseVectorDB`, `BaseGraphStore`, `BaseCacheBackend`). Service-provider ABCs live in the concern's
  own `base.py` (`services/embeddings/base.py`, `services/llm/base.py`, etc.).
- **Capability flags** live on the ABC. Example: `BaseVectorDB.supports_sparse: bool`. `dense_search`
  is always implemented; `sparse_search` raises `NotImplementedError` on stores that can't do it, and
  the retrieval service checks the flag and rejects/falls back — **no silent wrong answers**.
- **A factory is one small function**, not a class hierarchy:

```python
def create_vector_db() -> BaseVectorDB:
    p = get_settings().vector_store.provider
    if p == "qdrant":   return QdrantRepository(...)
    if p == "pgvector": return PgVectorRepository(...)
    if p == "chroma":   return ChromaRepository(...)
    raise ValueError(f"Unknown vector_store.provider: {p!r}")
```

**When to add a base class:** only for a concern with 2+ real implementations or config-selected
providers. One implementation → no base class, no factory (YAGNI). One base + flat concrete impls;
**no multi-tier inheritance.**

---

## 4. Directory & file map

```
app/
  main.py                         FastAPI app factory, router wiring, lifespan

  api/v1/routes/                  thin HTTP routes (prefix /api/v1)
    health.py                     GET /health
    documents.py                  list/get/chunks/delete documents
    ingestion.py                  ingest (async/sync/upload/by-filename/bytes), status
    query.py                      POST /query (one-shot, optionally agentic)
    chat.py                       chat turns + session/message management

  services/
    embeddings/   base.py + ollama.py · openai_provider.py · azure_openai.py + factory.py
    llm/          base.py + ollama_provider.py · openai_provider.py + factory.py
    parsing/      base.py + custom_parser.py · docling_client.py + factory.py
    chunking/     base.py + fixed.py · recursive.py · semantic.py + factory.py
    reranking/    base.py + bge.py + factory.py
    retrieval/
      service.py                  RetrievalService: mode dispatch, RRF fusion, graph retrieval
      pipeline.py                 QueryPipeline: embed→autofilter→retrieve→rerank→synthesize→cite
      autofilter.py               LLM-inferred metadata filters from the query
    ingestion/
      service.py                  IngestionService: register/dedup + full process pipeline + GraphRAG
    graph/
      service.py                  GraphRAGService: community detection + LLM reports, description merge
    agent/
      graph.py                    LangGraph state machine (analyze→retrieve→grade→loop)
      tool_agent.py               ReAct / deepagents tool-calling agent
    citation/
      builder.py                  build Citation (chunks, pages, confidence)

  repositories/
    base.py                       BaseVectorDB · BaseGraphStore · BaseCacheBackend (the infra ABCs)
    vector/      qdrant.py · pgvector.py · chroma.py + factory.py
    graph/
      _engine.py                  pure graph algorithms (cosine search, BFS, community match) — shared
      networkx_store.py           NetworkX backend: one JSON file per user (zero-infra default)
      age.py                      Apache AGE backend (Postgres); reads delegate to _engine
      factory.py
    cache/       ignite.py + factory.py
    relational/
      database.py                 async engine/session factory
      schema.py                   portable DDL + idempotent ALTER migrations (sqlite & postgres)
      documents.py · sessions.py · messages.py    repository functions (all take user_id)
      metadata_catalog.py         per-tenant attribute-key catalog (drives extraction + auto-filter)
    storage/
      files.py                    upload persistence under uploads/{user_id}/{version}/

  models/        common.py · document.py · ingestion.py · query.py · chat.py · agent.py
  prompts/       ingestion.py · retrieval.py · agent.py     (all LLM prompts as constants/builders)
  observability/ logging.py (Rich) · tracing.py (MLflow) · timing.py (StageTimer)
  utils/         config.py (typed pydantic-settings) · hashing.py · text.py (cap_text/cap_list)

config/config.yaml               single source of runtime configuration
```

---

## 5. The GraphRAG subsystem (deep dive)

This is the most involved part. It's a genuine GraphRAG (local **and** global), not just an
entity→chunk index.

**Storage** — `BaseGraphStore` with two backends sharing one algorithm engine:
- `networkx_store.py` (default) — per-user `{graph_dir}/{user_id}.json` graph + a
  `{user_id}.communities.json` sidecar. Entity vectors are stored inline (fine to ~10k entities/user;
  beyond that move to AGE/pgvector — see the `ponytail:` note at the file head).
- `age.py` — Apache AGE on Postgres. **Writes** are incremental Cypher; **reads** load the tenant's
  graph into memory and reuse `_engine.py`, so both backends behave identically.
  ⚠️ AGE is implemented but **not run-verified** in dev (no Postgres+AGE) — NetworkX is the tested path.
- `_engine.py` — pure functions on a `networkx.DiGraph`: `search_entities` (cosine), `expand_and_collect`
  (BFS, seeds-first ranking), `match_entities_by_name` (string fallback), `search_communities`,
  `communities_for_entities`, `merge_description`, `build_graph`.

**Data model:** entity node id = normalized name (so the same entity **merges across documents**);
node carries `name, type, description, embedding, chunks` (mentions encoded `"{document_id}\x1f{chunk_id}"`).
Relations are edges with `type, description, document_id`. Communities are stored reports
`{id, title, summary, members, embedding}`.

**Build (ingest)** — `services/ingestion/service.py::_extract_and_store_graph`:
1. Per chunk, one LLM call extracts entities + relations **with descriptions** (`ExtractedGraph`).
2. Accumulate across the document, embed one vector per unique entity (`name: description`).
3. **Batch** `upsert_entities` / `upsert_relations` (one write per document).
4. Communities (`GraphRAGService.rebuild_communities` — Louvain clustering → one LLM report per
   community → embed + store; powers global queries). Decoupled from ingest by default: it runs
   per-document only if `community_detection.rebuild_on_ingest` is on; otherwise trigger it on a
   schedule via `POST /documents/graph/rebuild-communities`.
5. Optional (`entity_extraction.summarize_descriptions`): LLM-collapse multi-source descriptions.

**Retrieve** — `services/retrieval/service.py`:
- `graph` (local): seed entities by **semantic** cosine (`search_entities`), falling back to string
  match; expand `graph_hops`; collect ranked chunks; **dual-level** — blend the seeds' community
  summaries in as `community`-tagged pseudo-chunks (`graph_dual_level`).
- `graph_global` (thematic): rank community reports by query similarity (`search_communities`),
  no entity anchoring.
- `hybrid_graph`: RRF-fuse dense + sparse + graph.

Config lives under `enrichment.entity_extraction`, `enrichment.community_detection`, and the
`retrieval.graph_*` keys.

---

## 6. Configuration

- `config/config.yaml` holds non-secret structure (providers, models, toggles, sizes).
- Typed into `pydantic-settings` models in `app/utils/config.py`; read everywhere via
  `get_settings()` (cached singleton).
- **Env overrides YAML:** prefix `RAG_`, nested keys via `__` (e.g. `RAG_LLM__API_KEY`,
  `RAG_DATABASE__DSN`). Secrets (API keys, DSNs) come from `.env` / environment only.

Each YAML block maps to one settings class. To add a knob: add the field (with a default + a
one-line comment) to the relevant `*Settings` class **and** mirror it in `config.yaml`.

---

## 7. How to add things (recipes)

### Add a provider (vector store, LLM, embedder, parser, chunker, reranker, graph store, cache)
1. Implement the concern's ABC in a **new file** next to the others
   (e.g. `app/repositories/vector/weaviate.py`), with `async` I/O methods and explicit type hints.
   Set any capability flags (e.g. `supports_sparse`).
2. Add **one branch** to that concern's `factory.py` mapping the new `provider` string to your class.
3. Add the provider's settings/fields to the matching `*Settings` class in `app/utils/config.py`
   and to `config/config.yaml`.
4. Leave a self-check (see §8). Every read/write **must** take and filter on `user_id`.

### Add an API route
1. Add the handler to the relevant file in `app/api/v1/routes/` (or a new file + `include_router` in
   `app/main.py`). Keep it thin — validate input into a model, call a service, return a model.
2. Define request/response **Pydantic models** in `app/models/`.
3. `user_id` comes from the **payload** (body field / form field / query param), default `"default"` —
   never a header, never trusted for scoping.

### Add a Pydantic model
Put it in the `app/models/*.py` file matching its domain (`query.py`, `ingestion.py`, `chat.py`,
`document.py`, `agent.py`). Structured **LLM outputs** are models too (used with
`llm.structured_output(..., schema=...)`).

### Add a retrieval mode
1. Add the name to `_VALID_MODES` in `services/retrieval/service.py`.
2. Add a branch in `RetrievalService.retrieve` (and a capability guard if it needs sparse/graph).
3. Tag results with `_tag(results, "yourmode")` for provenance. Add the mode to the
   `retrieval.default_mode` comment in config and docs.

### Add / change a prompt
All prompts live in `app/prompts/{ingestion,retrieval,agent}.py` as string constants or small
builder functions. **Never inline a prompt** in service code — import it from `app/prompts/`.

---

## 8. Project standards

These are enforced (see [CLAUDE.md](CLAUDE.md)). Optimize for **readability and maintainability**;
simple, linear, explicit code beats clever abstraction. When in doubt, write less.

**Code style**
- `async` on all I/O interface methods; explicit type hints on every signature and return type.
- Linear flow: minimize nesting, prefer early returns and explicit branches over clever one-liners
  or deep comprehensions.
- Self-documenting names over comments; comment only the non-obvious *why*.
- Absolute imports mapping to the layout: `from app.repositories.base import BaseVectorDB`.
- Pydantic for everything crossing a boundary.
- Fewest files that stay readable — don't split to split, don't pile five concerns into one.

**Abstraction** — add a base class only for a concern with 2+ implementations or config-selected
providers. One impl → no base, no factory. A factory is one function. No multi-tier inheritance.
Never abstract speculatively (YAGNI).

**Tenant isolation (never optional)** — every repository read and write takes `user_id` and filters
on it. Isolation is enforced at the repository layer; never trust the caller to scope the query.
Guard against path traversal in any file/graph path with `Path(component).name`.

**Exception handling** (standing convention — see the memory file of the same name):
- Specific exceptions first, then a general `except Exception`.
- **Non-breaking** failure → `logger.warning(...)` + `logger.debug(..., exc_info=True)`, then
  continue/degrade (e.g. ingest the document even if metadata extraction fails).
- **Breaking** failure → `logger.error/exception(..., exc_info=True)`.
- Repositories propagate; services/routes catch and degrade. Pure functions need no try/except.
- Declare custom exceptions in `app/exceptions.py` (create only when needed).

**Logging** — `app.*` loggers at the configured level; INFO for milestones, DEBUG for detail. At
DEBUG nothing is truncated. Use `cap_text` / `cap_list` (from `app/utils/text.py`) for previews
(`0` = no limit). Don't add new entries to `noisy_loggers` without reason.

**Tests** — non-trivial logic (a branch, loop, parser, money/security path) leaves **one** runnable
check: an `assert`-based `__main__` self-check or one small `test_*.py`. No frameworks or fixtures
unless asked. Trivial one-liners need no test.

---

## 9. Running self-checks

Modules with non-trivial logic carry a `__main__` self-check. Run them as modules:

```bash
python -m app.repositories.graph.networkx_store     # graph store: merge, traversal, communities, delete
python -m app.services.graph.service                # community build + description summarization
python -m app.services.retrieval.service            # query-entity selection + fallbacks
python -m app.repositories.relational.metadata_catalog
```

Quick import sweep (catches broken wiring across all modules):
```bash
python -c "import importlib,pkgutil,app; [importlib.import_module(m.name) for m in pkgutil.walk_packages(app.__path__,'app.')]; print('OK')"
```
