# Apache AGE — openCypher graph store on top of Postgres.
# Apache 2.0. Keeps vectors + graph + metadata in one ACID database.

import json
import logging
from typing import Any, Dict, List, Optional

from app.repositories.base import BaseGraphStore

logger = logging.getLogger(__name__)


class AgeRepository(BaseGraphStore):
    def __init__(self, dsn: str, graph_name: str = "rag_kg"):
        # asyncpg uses postgresql://, not postgresql+asyncpg://
        self._dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
        self.graph_name = graph_name
        self._pool = None

    async def _pool_or_create(self):
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(self._dsn)
            async with self._pool.acquire() as conn:
                await self._setup(conn)
        return self._pool

    async def _setup(self, conn) -> None:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS age")
        await conn.execute("LOAD 'age'")
        await conn.execute('SET search_path = ag_catalog, "$user", public')
        try:
            await conn.execute(f"SELECT create_graph('{self.graph_name}')")
        except Exception:
            pass  # graph already exists

    async def _run_cypher(
        self,
        conn,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        await conn.execute("LOAD 'age'")
        await conn.execute('SET search_path = ag_catalog, "$user", public')
        if params:
            rows = await conn.fetch(
                f"SELECT * FROM cypher('{self.graph_name}', $$ {cypher} $$, $1::agtype) AS (r agtype)",
                json.dumps(params),
            )
        else:
            rows = await conn.fetch(
                f"SELECT * FROM cypher('{self.graph_name}', $$ {cypher} $$) AS (r agtype)"
            )
        return [row["r"] for row in rows]

    async def upsert_entity(self, entity: Dict[str, Any], user_id: str) -> None:
        pool = await self._pool_or_create()
        entity_id = entity["id"]
        name = entity.get("name", entity_id)
        entity_type = entity.get("type", "Entity")
        props = {k: v for k, v in entity.get("properties", {}).items()}

        cypher = f"""
            MERGE (e:{entity_type} {{id: $entity_id, user_id: $user_id}})
            SET e.name = $name,
                e.document_id = $document_id,
                e.chunk_id = $chunk_id,
                e.updated_at = $updated_at
            RETURN e
        """
        params = {
            "entity_id": entity_id,
            "user_id": user_id,
            "name": name,
            "document_id": entity.get("document_id", ""),
            "chunk_id": entity.get("chunk_id", ""),
            "updated_at": "now",
        }
        async with pool.acquire() as conn:
            await self._run_cypher(conn, cypher, params)

    async def upsert_relation(self, relation: Dict[str, Any], user_id: str) -> None:
        pool = await self._pool_or_create()
        rel_type = relation.get("relation_type", "RELATED_TO").upper().replace(" ", "_")
        cypher = f"""
            MATCH (a {{id: $source_id, user_id: $user_id}})
            MATCH (b {{id: $target_id, user_id: $user_id}})
            MERGE (a)-[r:{rel_type} {{document_id: $document_id}}]->(b)
            RETURN r
        """
        params = {
            "source_id": relation["source_id"],
            "target_id": relation["target_id"],
            "user_id": user_id,
            "document_id": relation.get("document_id", ""),
        }
        async with pool.acquire() as conn:
            await self._run_cypher(conn, cypher, params)

    async def query(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        pool = await self._pool_or_create()
        async with pool.acquire() as conn:
            rows = await self._run_cypher(conn, cypher, params)
        return [json.loads(str(r)) if r else {} for r in rows]

    async def find_chunks_for_entities(self, entity_names: List[str], user_id: str) -> List[str]:
        """Return chunk_ids stored on entities that match the given names."""
        if not entity_names:
            return []
        pool = await self._pool_or_create()
        cypher = """
            MATCH (e {user_id: $user_id})
            WHERE e.chunk_id <> ''
              AND any(t IN $names WHERE toLower(e.name) CONTAINS t OR t CONTAINS toLower(e.name))
            RETURN e.chunk_id
        """
        params = {"user_id": user_id, "names": entity_names}
        async with pool.acquire() as conn:
            rows = await self._run_cypher(conn, cypher, params)

        # Each row is a single agtype scalar (a quoted chunk_id string). Dedupe, keep order.
        seen, chunk_ids = set(), []
        for r in rows:
            if r is None:
                continue
            chunk_id = json.loads(str(r))
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                chunk_ids.append(chunk_id)
        return chunk_ids

    async def delete_by_document(self, document_id: str, user_id: str) -> None:
        pool = await self._pool_or_create()
        cypher = """
            MATCH (e {document_id: $document_id, user_id: $user_id})
            DETACH DELETE e
        """
        async with pool.acquire() as conn:
            await self._run_cypher(conn, cypher, {"document_id": document_id, "user_id": user_id})
