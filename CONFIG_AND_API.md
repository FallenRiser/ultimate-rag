# Ultimate RAG — Config & API Reference

A decision-oriented guide: **what each config knob does, when to pick which value, and what every
API endpoint is for.** For the codebase map and standards see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

All config lives in `config/config.yaml`. Secrets/DSNs come from env (`.env`), prefix `RAG_`, nested
keys via `__` (e.g. `RAG_LLM__API_KEY`); **env overrides YAML**.

---

# Part A — Configuration

## Quick presets

| Goal | database | vector_store | graph_store | llm / embeddings | reranker |
|---|---|---|---|---|---|
| **Zero-infra (laptop, no Docker)** | `sqlite` | `chroma` | `networkx` | `ollama` | off |
| **Local + BM25/hybrid** | `sqlite` | `qdrant` | `networkx` | `ollama` | off |
| **Everything in one DB (ACID)** | `postgres` | `pgvector` | `age` | `openai`/`azure` | off |
| **Best quality (prod)** | `postgres` | `qdrant` | `age` | `openai` | `bge` on |

Rule of thumb: **`sqlite` + `chroma`/`qdrant` + `networkx`** for dev; **`postgres` + `pgvector`/`qdrant` + `age`** for scale/concurrency.

## `database` — relational store
- `provider: sqlite` → zero-infra file DB. Use for dev/single-process. **pgvector and AGE are NOT available** (they're Postgres-only), so pair with `chroma`/`qdrant` + `networkx`.
- `provider: postgres` → needed for `pgvector` and `age`. Use for multi-worker/production. Set `dsn` via env.

## `vector_store` — capability-aware
The active store decides which retrieval modes exist. Requesting BM25/hybrid on a semantic-only store is **rejected** (no FTS fake-out).

| provider | modes | infra | pick when |
|---|---|---|---|
| `chroma` | semantic only | none (local files) | zero-infra dev, semantic is enough |
| `pgvector` | semantic only | Postgres | you want vectors inside Postgres alongside relational + AGE |
| `qdrant` | semantic **+ BM25** | Qdrant server | you need `bm25`/`hybrid`/`hybrid_graph` (keyword recall) |

- `embedding_dim` **must match** your embedding model (`embeddings.dim`).
- `enable_sparse` (Qdrant) → turns on BM25 vectors. Needed for any sparse/hybrid mode.

## `graph_store` — knowledge graph (GraphRAG)
- `enabled: true` → entities/relations are extracted at ingest; enables `graph`/`hybrid_graph`/`graph_global` modes. Turning it on **is** the intent to fill it.
- `provider: networkx` → one JSON file per user, zero infra. Default. Single-process only (file + lock) — fine for dev/one worker.
- `provider: age` → Apache AGE on Postgres. Use for multi-worker/concurrency. (Implemented but not run-verified in dev — see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).)
- `graph_dir` (networkx) / `graph_name` (AGE) — storage location.

## `embeddings` — query/document vectors
- `provider: ollama` → local, zero-cost, no key. Default. `provider: openai`/`azure_openai` → hosted, higher quality, needs key.
- `model` + `dim` must agree with `vector_store.embedding_dim`. Changing the model means re-ingesting.
- `max_input_tokens` → caps text per embedding call to stay under the model's context (e.g. 512 for mxbai). `null` for big-context models.
- `batch_size` → embeddings per request; raise for throughput, lower if the server OOMs.

## `llm` — generation + all LLM extraction
- `provider: ollama` (local) | `openai` | `vllm` — all OpenAI-compatible. `tools`/`deep` agents need a **strong** model; small local models do fine for `graph` agent + extraction.
- `temperature: 0.1` for factual RAG; raise only for creative tasks.
- `max_tokens` → answer length ceiling. `timeout_s` / `max_retries` → resilience.

## `reranker` — precision booster
- `enabled: false` by default. Turn **on** when retrieval returns roughly-right-but-unordered results; it reorders the top candidates with a cross-encoder.
- `provider: bge` (default, via TEI `endpoint`, or local). `rerank_top_k` (in `retrieval`) caps what survives.

## `parsing`
- `backend: custom` → no-GPU, pure-Python (pypdf/docx/pptx/xlsx + OCR). Default; good for most docs.
- `backend: docling` → richer layout/table/figure extraction via docling-serve (HTTP). Use for complex PDFs, scanned docs, tables. Every `docling.*` knob is overridable **per request** via `parser_options`.
- `docling.do_ocr` / `force_ocr` → enable for scanned/image PDFs. Enrichment toggles (`do_table_structure`, `do_picture_description`, …) add quality at latency/cost.

## `chunking`
- `strategy: recursive` (default) → structure-aware splits. `fixed` → simplest. `semantic` → splits on embedding similarity (`semantic_threshold`), best coherence, more compute. `document_aware` → respects headings/sections.
- `chunk_size` (tokens) should align with `embeddings.max_input_tokens`. `chunk_overlap` ~15% preserves context across boundaries.

## `enrichment` — all optional, ingest-time
| Block | What it does | Turn on when | Cost |
|---|---|---|---|
| `metadata_extraction` | LLM extracts title/author/doc_type + free-form attributes → **filterable** | you want metadata filters / auto-filter | 1 LLM call/doc |
| `entity_extraction` | entities + relations + descriptions for the graph | `graph_store.enabled` (runs automatically) | 1 LLM call/chunk |
| `entity_extraction.embed_entities` | embed entities → **semantic** query→entity matching | keep on for good graph recall | embeddings/doc |
| `entity_extraction.summarize_descriptions` | LLM-merge an entity's descriptions across docs | quality matters, cost OK | LLM/merged entity |
| `community_detection` | Louvain clusters + LLM reports → **global/thematic** queries | you ask "what are the main themes" questions | expensive (see below) |
| `section_metadata` | per-chunk attributes (section/clause/price) | structured docs (laws, contracts) | 1 LLM call/chunk |
| `table_extraction` / `image_captioning` / `contextual_retrieval` | table summaries / captions / per-chunk doc-context prefix | content-specific quality wins | varies |

- `*.max_chunks` / `*.max_chars`: `0` = no limit. Lower them to cap LLM cost on big documents.
- **`community_detection.rebuild_on_ingest`**: `false` (default) → don't re-cluster on every ingest; trigger rebuilds via `POST /documents/graph/rebuild-communities` on a schedule. `true` → auto-rebuild per doc (small corpora only — it re-clusters the **whole** tenant graph).

## `retrieval` — modes & GraphRAG knobs
**`default_mode`** (override per request with `mode`). When to use which:

| mode | what it does | use when | requires |
|---|---|---|---|
| `semantic` | dense vector search | meaning-based lookup; always available | — |
| `bm25` | keyword/sparse search | exact terms, codes, names | qdrant |
| `hybrid` | dense + BM25, RRF-fused | best general default for keyword+meaning | qdrant |
| `graph` | local GraphRAG: seed entities → multi-hop → chunks (+ dual-level community context) | questions about specific entities & how they relate | graph_store |
| `hybrid_graph` | dense + BM25 + graph, fused | maximum recall | qdrant + graph_store |
| `graph_global` | rank **community reports** (thematic) | "main themes / summarize everything about X" | graph_store (+ communities; else falls back) |

- `top_k` → candidates retrieved. `rerank_top_k` → kept after rerank.
- `fusion: rrf` (rank-based, robust) | `weighted` (uses `weights.dense/sparse`).
- `graph_hops` → relation edges traversed from matched entities (0 = direct mentions only; 1–2 typical).
- `graph_query_entities: regex` (cheap, no LLM) | `llm` (one extra call, better entity recall).
- `graph_seed_top_k` → how many entities seed local graph search. `graph_global_top_k` → community reports pulled in global mode.
- `graph_dual_level` → local `graph` mode also blends the seeds' community summaries (needs communities built).

## `agent`
- `enabled` → route `/query` and `/chat` through an agent (vs. the plain pipeline). Override per request with `use_agent`.
- **`style`** — the big choice:
  - `graph` → fixed LangGraph state machine (analyze → retrieve → grade → loop). **Works with small/local LLMs.** Default.
  - `tools` → ReAct tool-calling; the model decides when/how to search. Needs a **strong** LLM.
  - `deep` → deepagents planner + sub-agents. Strongest LLM; complex multi-step questions.
- `query_rewrite` / `decompose` / `max_subqueries` → break hard questions into sub-queries.
- `grade_relevance` + `max_retrieval_loops` → re-retrieve if graded weak. `grade_max_chunks` caps grader cost.
- `auto_filter` → LLM infers metadata filters from the query (e.g. "in section 302" → `section=302`). `fields` = core fields allowed; `use_catalog` = also discovered attribute keys; `max_catalog_keys` caps them.

## `cache`, `logging`, `observability`, `ingestion`, `chat`
- `cache.enabled` → cache identical (query+filters+user+mode) answers in Ignite for `ttl_seconds`. Turn on for repeated queries.
- `logging.level: DEBUG` → nothing truncated (full prompts/chunks). `preview_chars` caps log previews (`0` = full). Production: `INFO`.
- `observability.mlflow.enabled` → trace every request (one span tree). `max_chunk_chars` caps chunk text in traces.
- `ingestion.max_file_mb`, `dedup_by_hash` (skip re-ingesting identical content), `storage_dir`.
- `chat.memory_enabled`, `max_turns` (history turns fed to the agent).

---

# Part B — API (prefix `/api/v1`)

Interactive docs at `/docs`. **`user_id` is always in the payload** (body field, form field, or query
param), never a header; absent → `"default"`; every repository filters on it.

## Health
**`GET /health`** — liveness probe. No auth, no params.

## Ingestion
All return `IngestionResponse {document_id, task_id, status}` (`status`: `queued` | `ready` | `failed` | `duplicate`).

| Endpoint | Body / params | What it does · when to use |
|---|---|---|
| **`POST /ingest`** | multipart: `file`, `user_id`, `metadata?` (JSON), `parser_options?` (JSON) | **Async** — registers + processes in a background task, returns `queued` immediately. Default ingest path. Poll status. |
| **`POST /ingest/sync`** | same as above | **Inline** — runs the full pipeline and returns the final `ready`/`failed`. Use for small files or when you need the result now (scripts, tests). |
| **`POST /ingest/upload`** | multipart: `version`, `files[]`, `user_id` | Save raw files under `{storage_dir}/{user_id}/{version}/` **without** ingesting. Step 1 of a two-step flow. Returns `UploadedFile[]`. |
| **`POST /ingest/by-filename`** | JSON: `version`, `filenames[]`, `user_id`, `sync?`, `metadata?`, `parser_options?` | Ingest files already uploaded (step 2). `sync:true` runs inline. |
| **`POST /ingest/bytes`** | JSON: `files[{filename, content_base64}]`, `user_id`, `sync?`, `metadata?`, `parser_options?` | Ingest raw bytes directly, no storage. Use from code without multipart. |
| **`GET /ingest/status/{document_id}`** | query: `user_id` | Ingestion progress (`0.0` queued → `0.5` processing → `1.0` ready). |

- `metadata` → JSON object; keys become **filterable** at query time (caller-supplied wins on conflict).
- `parser_options` → JSON object overriding `parsing.docling.*` for this document only.

## Documents
| Endpoint | What it does |
|---|---|
| **`GET /documents`** (`user_id`) | list the tenant's documents |
| **`GET /documents/{id}`** (`user_id`) | one document's metadata/status |
| **`GET /documents/{id}/chunks`** (`user_id`) | the document's stored chunks |
| **`DELETE /documents/{id}`** (`user_id`) | delete the document — **cascades** its relational row, vectors, and graph entities/relations |
| **`POST /documents/graph/rebuild-communities`** (`user_id`) | re-cluster the tenant graph + regenerate community reports (the `graph_global` layer). Run on a schedule. Needs `graph_store.enabled` + `community_detection.enabled`; returns `{communities: N}`. |

## Query — `POST /query`
One-shot RAG. Body (`QueryRequest`):

| field | default | effect |
|---|---|---|
| `query` | — | the question (required) |
| `user_id` | `"default"` | tenant isolation |
| `mode` | config `default_mode` | force a retrieval mode (see table above) |
| `filters` | `{}` | exact-match metadata filters (e.g. `{"doc_type":"contract"}`); **wins over** auto-filter |
| `top_k` | config `top_k` | candidates to retrieve |
| `use_agent` | config `agent.enabled` | `true`/`false` to force agentic vs. plain pipeline |
| `agent_style` | config `agent.style` | `graph` \| `tools` \| `deep` |
| `session_id` | — | echoed back; no memory (use `/chat` for memory) |

Returns `QueryResponse {answer, citations{source_chunks, confidence}, mode_used, app_timings}`.
Errors: `400` bad request/mode, `501` unsupported, `500` internal.

**When to use what:** `use_agent:false` = fastest, single retrieval. `agent_style:graph` = robust default, small-LLM-friendly. `tools`/`deep` = hard multi-step questions with a strong LLM.

## Chat — stateful
| Endpoint | What it does |
|---|---|
| **`POST /chat`** | one turn with memory. Body (`ChatRequest`): `message`, `user_id`, `session_id?` (omit to start a new session), `filters?`, `agent_style?`. Loads prior turns, resolves follow-ups, persists both messages + citations. Returns `ChatResponse {session_id, answer, message, citations, app_timings}`. |
| **`GET /chat/sessions`** | `user_id`, `page`, `page_size` — paginated session list (`PaginatedResponse`) |
| **`GET /chat/sessions/{id}/messages`** | `user_id`, `page`, `page_size` — paginated message history (chronological) |
| **`DELETE /chat/sessions/{id}`** | `user_id` — delete a session (**cascades** its messages) |

Use **`/chat`** when you need conversational memory/follow-ups; use **`/query`** for stateless one-shots.
