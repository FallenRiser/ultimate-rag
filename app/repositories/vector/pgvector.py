import json
from typing import Any, Dict, List, Optional

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.repositories.base import BaseVectorDB


def _metadata_filter_sql(filters: Optional[Dict[str, Any]], params: Dict[str, Any]) -> str:
    """Build `AND metadata ->> :fkN ...` clauses. A scalar matches exactly; a list matches ANY
    of its values (OR within the key, via IN). Keys/values bound as params (no injection)."""
    if not filters:
        return ""
    clauses = ""
    for i, (key, value) in enumerate(filters.items()):
        params[f"fk{i}"] = key
        if isinstance(value, (list, tuple, set)):
            placeholders = []
            for j, item in enumerate(value):
                params[f"fv{i}_{j}"] = str(item)
                placeholders.append(f":fv{i}_{j}")
            if placeholders:
                clauses += f" AND metadata ->> :fk{i} IN ({', '.join(placeholders)})"
        else:
            params[f"fv{i}"] = str(value)
            clauses += f" AND metadata ->> :fk{i} = :fv{i}"
    return clauses


class PgvectorRepository(BaseVectorDB):
    supports_sparse = False  # pgvector: semantic only — no BM25

    def __init__(self, engine: AsyncEngine, table: str = "chunk_vectors"):
        self.engine = engine
        self.table = table

    async def initialize_collection(self, vector_size: int) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id          TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    version_id  TEXT NOT NULL,
                    user_id     TEXT NOT NULL,
                    text        TEXT NOT NULL,
                    embedding   vector({vector_size}),
                    metadata    JSONB DEFAULT '{{}}'::jsonb
                )
            """))
            await conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS {self.table}_hnsw_idx
                ON {self.table} USING hnsw (embedding vector_cosine_ops)
            """))
            await conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS {self.table}_user_idx ON {self.table} (user_id)
            """))

    async def insert(self, chunks: List[Dict[str, Any]], user_id: str) -> bool:
        async with self.engine.begin() as conn:
            for chunk in chunks:
                chunk["metadata"]["user_id"] = user_id
                await conn.execute(text(f"""
                    INSERT INTO {self.table}
                        (id, document_id, version_id, user_id, text, embedding, metadata)
                    VALUES
                        (:id, :document_id, :version_id, :user_id, :text,
                         CAST(:embedding AS vector), CAST(:metadata AS jsonb))
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata  = EXCLUDED.metadata
                """), {
                    "id": chunk["chunk_id"],
                    "document_id": chunk["metadata"]["document_id"],
                    "version_id": chunk["metadata"]["version_id"],
                    "user_id": user_id,
                    "text": chunk["text"],
                    "embedding": str(chunk["dense_vector"]),
                    "metadata": json.dumps(chunk["metadata"]),
                })
        return True

    async def dense_search(
        self,
        query_vector: List[float],
        user_id: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        params = {"qv": str(query_vector), "user_id": user_id, "top_k": top_k}
        filter_sql = _metadata_filter_sql(filters, params)
        async with self.engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT id, document_id, version_id, text, metadata,
                       1 - (embedding <=> CAST(:qv AS vector)) AS score
                FROM {self.table}
                WHERE user_id = :user_id{filter_sql}
                ORDER BY embedding <=> CAST(:qv AS vector)
                LIMIT :top_k
            """), params)
            rows = result.fetchall()
        return [
            {
                "id": row.id,
                "score": float(row.score),
                "payload": {
                    "text": row.text,
                    "metadata": row.metadata,
                    "document_id": row.document_id,
                    "version_id": row.version_id,
                },
            }
            for row in rows
        ]

    async def fetch_by_ids(
        self, chunk_ids: List[str], user_id: str
    ) -> List[Dict[str, Any]]:
        if not chunk_ids:
            return []
        stmt = text(f"""
            SELECT id, document_id, version_id, text, metadata
            FROM {self.table}
            WHERE user_id = :user_id AND id IN :ids
        """).bindparams(bindparam("ids", expanding=True))
        async with self.engine.connect() as conn:
            result = await conn.execute(stmt, {"user_id": user_id, "ids": list(chunk_ids)})
            rows = result.fetchall()
        return [
            {
                "id": row.id,
                "score": 1.0,
                "payload": {
                    "text": row.text,
                    "metadata": row.metadata,
                    "document_id": row.document_id,
                    "version_id": row.version_id,
                },
            }
            for row in rows
        ]

    async def list_by_document(
        self, document_id: str, user_id: str
    ) -> List[Dict[str, Any]]:
        async with self.engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT id, document_id, version_id, text, metadata
                FROM {self.table}
                WHERE document_id = :doc AND user_id = :user_id
            """), {"doc": document_id, "user_id": user_id})
            rows = result.fetchall()
        return [
            {
                "id": row.id,
                "payload": {
                    "text": row.text,
                    "metadata": row.metadata,
                    "document_id": row.document_id,
                    "version_id": row.version_id,
                },
            }
            for row in rows
        ]

    async def delete_by_document(self, document_id: str, user_id: str) -> bool:
        async with self.engine.begin() as conn:
            await conn.execute(text(f"""
                DELETE FROM {self.table} WHERE document_id = :doc_id AND user_id = :user_id
            """), {"doc_id": document_id, "user_id": user_id})
        return True

    async def delete_by_version(self, version_id: str, user_id: str) -> bool:
        async with self.engine.begin() as conn:
            await conn.execute(text(f"""
                DELETE FROM {self.table} WHERE version_id = :version_id AND user_id = :user_id
            """), {"version_id": version_id, "user_id": user_id})
        return True
