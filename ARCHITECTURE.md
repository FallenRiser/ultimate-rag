# Ultimate RAG — Architecture & Build Plan

An enterprise-grade, **agentic**, multi-retrieval RAG platform. Every external
dependency sits behind a small abstract base class selected via `config/config.yaml`,
so swapping a vector DB / LLM / embedder / parser / cache is a one-file change.

---

## 1. Locked decisions

| Area | Decision | Rationale |
|---|---|---|
| Agent orchestration | **LangGraph** | Explicit state-graph (decompose→route→retrieve→grade→loop), built-in Postgres checkpointer = chat memory, MLflow autolog. |
| Vector store | `BaseVectorDB` ABC → **pgvector** (semantic) + **Chroma** (semantic) + **Qdrant** (semantic + BM25) | **Capability-aware:** the active store decides available modes. `supports_sparse` flag gates BM25/hybrid. |
| Relational | Postgres or SQLite, config-selected via `database.provider` | SQLite = zero-infra dev; Postgres needed for pgvector/AGE. |
| Sparse / BM25 | Qdrant sparse vectors (FastEmbed BM25) + Query API RRF | pgvector has no BM25 → semantic-only. No FTS bolt-on to fake it. |
| Knowledge graph | **Deferred to Phase 2.** `GraphStore` ABC scaffolded now; **Apache AGE** is the planned target | Apache 2.0, keeps graph on Postgres. Neo4j Community is GPLv3 (excluded); Kuzu abandoned Oct 2025. |
| Embeddings | `EmbeddingProvider` ABC → Azure OpenAI + OpenAI + Ollama | OpenAI-compatible path covers most future APIs. |
| LLMs | `LLMProvider` ABC → OpenAI (base_url) + Ollama + vLLM | All OpenAI-compatible → mostly one adapter + config. |
| Doc extraction | `DocumentParser` ABC → Docling-serve client **and** custom (pypdf/python-docx/python-pptx/openpyxl/OCR) | Custom is the no-GPU fallback; backend chosen in config. |
| Chunking | `Chunker` ABC → fixed / recursive / document-aware / semantic | Falls back to vanilla when structure unavailable. |
| Enrichment | Toggleable steps: image captioning, table extraction, metadata extraction, contextual prefixing | Each on/off in config. |
| Query modes | `semantic` / `bm25` / `graph` / `hybrid` / `hybrid_graph` | Per request; **validated against the active store's capabilities** (e.g. bm25/hybrid rejected on pgvector). |
| Versioning | Content-hash + immutable `document_versions`; re-embed on change | Rollback + audit. |
| Observability | **MLflow Tracing** (autolog + manual spans) + **Rich** logging | Self-hosted; DEBUG = full untruncated traces. |
| Ingestion | **In-process** — FastAPI background task (`POST /ingest`) or inline (`POST /ingest/sync`) | No broker/worker to operate; bytes processed in-memory. |
| Metadata | Structured columns + JSONB; optional LLM extraction at ingest; filterable at query | Highest-ROI enterprise feature. |
| Structured output | **instructor** + Pydantic models, auto-retry on validation fail | Matches requirement to parse LLM outputs with Pydantic. |
| Rerankers | `Reranker` ABC → BGE-reranker-v2-m3 (default, via TEI), Qwen3 / Cohere optional | BGE = lightweight safe default. |
| Chat memory | LangGraph Postgres checkpointer + session store | Per-session memory. |
| User isolation | `user_id` enforced as a mandatory filter at the **repository** layer | Never trusted from the caller. |
| Citations | Return `answer + source_chunks + page_numbers + confidence` | Confidence from rerank score + grounding check. |
| Query cache | `CacheBackend` ABC → **Apache Ignite** (default), Redis later | Key = normalized(query)+filters+user+mode. |

### Deferred with intent
- **DSPy** (prompt optimization) — Phase 3. Prompts are kept as standalone modules now so DSPy/MIPRO drops in once an eval set exists.
- **Haystack** — not used as the backbone; it fights the fine-grained customization this spec wants.

---

## 2. Layered architecture

```
                         ┌──────────────────────────────────────────┐
   HTTP (FastAPI)  ─────► │  app/api/v1/routes  (thin, no logic)      │
                         └───────────────┬──────────────────────────┘
                                         │ Pydantic models (app/models)
                         ┌───────────────▼──────────────────────────┐
   Agent (LangGraph) ───►│  app/services  (all business logic)        │
   rewrite→route→        │   ingestion │ retrieval │ agent │ citation │
   retrieve→rerank→      └───────────────┬──────────────────────────┘
   grade→synthesize→cite                 │ ABC + factory (config-driven)
                         ┌───────────────▼──────────────────────────┐
   Providers / DBs  ────►│ embeddings │ llm │ parsing │ chunking │    │
                         │ reranking  │ vector │ graph │ cache        │
                         └───────────────┬──────────────────────────┘
                                         │
       Postgres (pgvector + AGE + relational) / SQLite • Qdrant • Chroma • Ignite

   Cross-cutting:  MLflow tracing • Rich logging • X-User-Id isolation • config
```

### Folder map (matches the required layout)
```
app/
  main.py                     FastAPI entrypoint + lifespan wiring
  api/v1/routes/              one file per route group (health, ingestion, query, chat, documents)
  api/deps.py                 auth / current-user / DI helpers
  observability/              logging (Rich) + tracing (MLflow)
  services/                   business logic, small files, grouped by concern
    embeddings/ llm/ parsing/ chunking/ reranking/
    retrieval/ ingestion/ agent/ citation/
  utils/                      config loader, registry, hashing, text helpers
  models/                     Pydantic models (DTOs + domain)
  repositories/               DB connections: vector/ graph/ cache/ relational/
config/config.yaml            single source of runtime configuration
logs/*.log                    Rich file logs
```

**Provider pattern.** Each concern has: `base.py` (ABC), one file per implementation,
and `factory.py` (a registry keyed by the `provider` string in config). Services depend
only on the ABC; the factory resolves the concrete class at startup.

---

## 3. Agentic retrieval flow (LangGraph)

```
START
  └─ analyze_query        classify intent, choose retrieval mode, detect filters
  └─ rewrite/decompose    HyDE / multi-query / sub-question split (configurable)
  └─ route                semantic | bm25 | graph | hybrid | hybrid_graph
  └─ retrieve (×N, parallel per sub-query)   + mandatory user_id filter
  └─ fuse                 RRF / weighted across modes
  └─ rerank               cross-encoder top-k
  └─ grade                relevance/grounding check → loop back if weak (max N)
  └─ synthesize           answer with inline citations (structured output)
  └─ cite                 build CitationBundle (chunks, pages, confidence)
END
```
State is a Pydantic model. Chat memory = LangGraph checkpointer (Postgres) keyed by `session_id`.

---

## 4. Data model (relational, Postgres)

- `documents` — id, user_id, source, mime, status, created_at, current_version_id
- `document_versions` — id, document_id, version_no, content_hash, created_at (immutable)
- `chunks` — id, document_id, version_id, ordinal, text, page, metadata(JSONB), token_count
- `chat_sessions` — id, user_id, title, created_at
- `chat_messages` — id, session_id, role, content, citations(JSONB), created_at
- Vectors live in pgvector/Qdrant tagged with `{user_id, document_id, version_id, chunk_id}`.
- LangGraph checkpoints in their own table (managed by the checkpointer).

**Isolation:** every retrieval injects `user_id` as a hard filter at the repository layer
(vector payload filter + SQL WHERE + Cypher WHERE). Callers cannot bypass it.

---

## 5. Configuration & secrets
- `config/config.yaml` holds non-secret structure (providers, models, toggles, sizes).
- Secrets (API keys, DSNs) come from environment / `.env` and override YAML.
- Loaded once into typed `pydantic-settings` models (`app/utils/config.py`); injected via DI.

---

## 6. Observability & logging
- **Rich** console + rotating file logs in `logs/`. At `DEBUG`, **nothing is truncated**:
  full query, retrieval candidates, the exact prompt + params sent to the LLM, and the raw
  LLM response are all logged.
- **MLflow Tracing**: autolog for LangGraph/LangChain + LLM/embedding clients, plus manual
  spans around ingestion and retrieval stages. OTel-compatible export to enterprise stacks.

---

## 7. Phased roadmap

**Phase 0 — Skeleton (this deliverable):** folder structure, all base classes + factories,
config schema, Pydantic models, observability bootstrap, API route stubs. Importable, runnable, no business logic.

**Phase 1 — Core RAG:** custom + Docling parsing, chunking strategies, Azure/Ollama embeddings,
pgvector + Qdrant stores, semantic/BM25/hybrid retrieval, citations, user isolation, async
ingestion (ARQ), MLflow + Rich, metadata + filters, document versioning.

**Phase 2 — Agentic + graph + scale:** LangGraph agent (rewrite/decompose/grade/loop),
reranking, Apache AGE knowledge graph + graph/hybrid_graph modes, chat sessions w/ memory,
Apache Ignite query cache, enrichment (captions/tables/contextual).

**Phase 3 — Optimization & hardening:** eval harness (Ragas), DSPy prompt optimization,
guardrails/PII, rate limiting + circuit breakers, re-embed jobs on model change.

---

## 8. Suggested additions (beyond the 25 requirements)
- **Eval harness first** (Ragas) — prerequisite for DSPy and for measuring any change.
- **Contextual retrieval** (chunk-context prefixing) — large quality win, cheap to add.
- **Auth (JWT/OIDC)** in front of the `X-User-Id` isolation layer (no auth today — the header is trusted; put a gateway/IdP in front for production).
- **Ingestion idempotency/dedup** via content hash.
- **Provider resilience** — retries, timeouts, circuit breakers around LLM/embedding calls.
- **Alembic migrations** for the relational schema.
