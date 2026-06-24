# Ultimate RAG — Project Conventions

The standard for all code in this repo. **Optimize for readability and maintainability.**
Simple, linear, explicit code beats clever abstraction. When in doubt, write less.

## Architecture (layered, one job per layer)
- `app/api/v1/routes/` — thin HTTP routes, **no business logic**
- `app/services/` — business logic, small files grouped by concern
- `app/repositories/` — infrastructure access (DBs, caches, queues)
- `app/models/` — Pydantic models (validation/serialization at boundaries)
- `app/observability/` · `app/utils/` · `config/`

Business logic talks to **base classes (interfaces)**, never concrete vendor classes.

## When to abstract — and when not to
- Add a base class **only** for a concern with 2+ real implementations or config-selected
  providers: vector store, graph store, LLM, embeddings, parser, chunker, reranker, cache.
- One implementation → no base class, no factory. Never abstract speculatively (YAGNI).
- A "factory" is **one small function** mapping the config `provider` string to a class —
  not a factory-class hierarchy.
- One base + flat concrete impls. **No multi-tier inheritance.**

## Code style
- `async` methods on all I/O interfaces.
- Explicit type hints on every signature and return type.
- Linear flow: minimize nesting, prefer early returns and explicit branches over clever
  one-liners or deep comprehensions.
- Self-documenting names over comments. Comment only the non-obvious *why*.
- Absolute imports mapping to the layout: `from app.repositories.base import BaseVectorDB`.
- Pydantic for everything crossing a boundary (request/response, LLM output).
- Fewest files that stay readable. Don't split a file just to split it; don't pile five
  concerns into one either.

## Canonical example — match this readability
```python
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue,
)
from typing import Any, Dict, List
from app.repositories.base import BaseVectorDB


class QdrantRepository(BaseVectorDB):
    supports_sparse = True  # qdrant does semantic + BM25

    def __init__(self, url: str, api_key: str, collection_name: str = "enterprise_knowledge"):
        self.client = AsyncQdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

    async def initialize_collection(self, vector_size: int) -> None:
        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                sparse_vectors_config={"sparse": {}},
            )

    async def dense_search(
        self, query_vector: List[float], tenant_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        tenant_filter = Filter(
            must=[FieldCondition(key="metadata.tenant_id", match=MatchValue(value=tenant_id))]
        )
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=tenant_filter,
            limit=top_k,
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]
```

## Capability-aware databases (important)
The **active vector store decides which retrieval modes exist**:
- `pgvector` → **semantic only** (no BM25/sparse).
- `qdrant` → **semantic + BM25** (hybrid).

Mechanism: the base class declares `supports_sparse: bool`. `dense_search` is always
implemented; `sparse_search` raises `NotImplementedError` on stores that can't do it.
The retrieval service checks `supports_sparse` and **rejects (or falls back to semantic)**
when a BM25/hybrid mode is requested on a store that doesn't support it. No silent wrong
answers, no FTS bolt-on to fake BM25 on pgvector.

## User / tenant isolation (never optional)
Every repository read and write takes `tenant_id` and filters on it. Isolation is enforced
at the repository layer — never trust the caller to scope the query.

## Tests
Non-trivial logic (a branch, loop, parser, money/security path) leaves **one** runnable
check: an `assert`-based `__main__` self-check or one small `test_*.py`. No frameworks or
fixtures unless asked. Trivial one-liners need no test.
