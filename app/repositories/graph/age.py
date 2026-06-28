# Apache AGE — openCypher graph store on top of Postgres. Apache 2.0; keeps vectors + graph +
# metadata in one ACID database. Writes are incremental Cypher; reads load the tenant's graph
# into memory and reuse the shared algorithms in app.repositories.graph._engine (same code path
# as the NetworkX store), which keeps the Cypher simple and the two backends behaviourally equal.
#
# Not runnable-verified here (needs Postgres + AGE); the NetworkX store is the tested default.

import json
import logging
from typing import Any, Dict, List, Optional

import networkx as nx

from app.repositories.base import BaseGraphStore
from app.repositories.graph import _engine
from app.utils.config import get_settings

logger = logging.getLogger(__name__)

_SEP = _engine.SEP


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
        self, conn, cypher: str, params: Optional[Dict[str, Any]] = None
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

    async def _fetch_maps(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run a query whose RETURN is a single map literal per row; parse to plain dicts."""
        pool = await self._pool_or_create()
        async with pool.acquire() as conn:
            rows = await self._run_cypher(conn, cypher, params)
        out = []
        for r in rows:
            if r is None:
                continue
            value = json.loads(str(r))
            if isinstance(value, dict):
                out.append(value)
        return out

    async def _load_graph(self, user_id: str) -> nx.DiGraph:
        entities = await self._fetch_maps(
            """MATCH (e:Entity {user_id: $user_id})
               RETURN {id: e.id, name: e.name, type: e.type, description: e.description,
                       chunks: e.chunks, embedding: e.embedding}""",
            {"user_id": user_id},
        )
        relations = await self._fetch_maps(
            """MATCH (a:Entity {user_id: $user_id})-[r]->(b:Entity {user_id: $user_id})
               RETURN {source: a.id, target: b.id, type: type(r), description: r.description}""",
            {"user_id": user_id},
        )
        return _engine.build_graph(entities, relations)

    async def _load_communities(self, user_id: str) -> List[Dict[str, Any]]:
        return await self._fetch_maps(
            """MATCH (c:Community {user_id: $user_id})
               RETURN {id: c.id, title: c.title, summary: c.summary,
                       members: c.members, embedding: c.embedding}""",
            {"user_id": user_id},
        )

    # --- writes -------------------------------------------------------------
    async def upsert_entities(self, entities: List[Dict[str, Any]], user_id: str) -> None:
        if not entities:
            return
        cap = get_settings().enrichment.entity_extraction.description_max_chars
        cypher = """
            MERGE (e:Entity {id: $entity_id, user_id: $user_id})
            SET e.name = $name, e.type = $type, e.embedding = $embedding,
                e.chunks = CASE
                    WHEN $mention = '' OR $mention IN coalesce(e.chunks, []) THEN coalesce(e.chunks, [])
                    ELSE coalesce(e.chunks, []) + $mention END,
                e.description = CASE
                    WHEN $description = '' THEN coalesce(e.description, '')
                    WHEN coalesce(e.description, '') = '' THEN $description
                    ELSE substring(coalesce(e.description, '') + ' | ' + $description, 0, $cap) END
        """
        pool = await self._pool_or_create()
        async with pool.acquire() as conn:
            for entity in entities:
                chunk_id = entity.get("chunk_id", "")
                mention = f'{entity.get("document_id", "")}{_SEP}{chunk_id}' if chunk_id else ""
                await self._run_cypher(conn, cypher, {
                    "entity_id": entity["id"], "user_id": user_id,
                    "name": entity.get("name", entity["id"]), "type": entity.get("type", "Entity"),
                    "embedding": entity.get("embedding"), "mention": mention,
                    "description": entity.get("description", ""), "cap": cap,
                })
        logger.debug("Graph(AGE): upserted %d entity mention(s) for user %s", len(entities), user_id)

    async def upsert_relations(self, relations: List[Dict[str, Any]], user_id: str) -> None:
        if not relations:
            return
        pool = await self._pool_or_create()
        async with pool.acquire() as conn:
            for relation in relations:
                rel_type = relation.get("relation_type", "related_to").upper().replace(" ", "_")
                cypher = f"""
                    MATCH (a:Entity {{id: $source_id, user_id: $user_id}})
                    MATCH (b:Entity {{id: $target_id, user_id: $user_id}})
                    MERGE (a)-[r:{rel_type} {{document_id: $document_id}}]->(b)
                    SET r.description = $description
                """
                await self._run_cypher(conn, cypher, {
                    "source_id": relation["source_id"], "target_id": relation["target_id"],
                    "user_id": user_id, "document_id": relation.get("document_id", ""),
                    "description": relation.get("description", ""),
                })
        logger.debug("Graph(AGE): upserted %d relation(s) for user %s", len(relations), user_id)

    async def update_entity_descriptions(self, descriptions: Dict[str, str], user_id: str) -> None:
        if not descriptions:
            return
        pool = await self._pool_or_create()
        cypher = "MATCH (e:Entity {id: $id, user_id: $user_id}) SET e.description = $text"
        async with pool.acquire() as conn:
            for node_id, text in descriptions.items():
                await self._run_cypher(conn, cypher, {"id": node_id, "user_id": user_id, "text": text})

    async def save_communities(self, communities: List[Dict[str, Any]], user_id: str) -> None:
        pool = await self._pool_or_create()
        async with pool.acquire() as conn:
            await self._run_cypher(conn, "MATCH (c:Community {user_id: $user_id}) DETACH DELETE c",
                                   {"user_id": user_id})
            for community in communities:
                await self._run_cypher(conn, """
                    CREATE (c:Community {user_id: $user_id, id: $id, title: $title,
                                         summary: $summary, members: $members, embedding: $embedding})
                """, {
                    "user_id": user_id, "id": community["id"], "title": community.get("title", ""),
                    "summary": community.get("summary", ""), "members": community.get("members", []),
                    "embedding": community.get("embedding"),
                })
        logger.debug("Graph(AGE): saved %d community report(s) for user %s", len(communities), user_id)

    # --- reads (delegate to the shared engine) ------------------------------
    async def search_entities(
        self, query_vector: List[float], user_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        results = _engine.search_entities(await self._load_graph(user_id), query_vector, top_k)
        logger.debug("Graph(AGE): entity search for user %s -> %d hit(s)", user_id, len(results))
        return results

    async def match_entities_by_name(self, names: List[str], user_id: str) -> List[Dict[str, Any]]:
        results = _engine.match_entities_by_name(await self._load_graph(user_id), names)
        logger.debug("Graph(AGE): name match %s for user %s -> %d hit(s)", names, user_id, len(results))
        return results

    async def expand_and_collect(self, seed_ids: List[str], user_id: str, hops: int) -> List[str]:
        if not seed_ids:
            return []
        chunk_ids = _engine.expand_and_collect(await self._load_graph(user_id), seed_ids, hops)
        logger.debug("Graph(AGE): expand %d seed(s) (%d hops) for user %s -> %d chunk(s)",
                     len(seed_ids), hops, user_id, len(chunk_ids))
        return chunk_ids

    async def load_graph_data(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        graph = await self._load_graph(user_id)
        entities = [
            {"id": n, "name": a.get("name", n), "type": a.get("type", "Entity"),
             "description": a.get("description", "")}
            for n, a in graph.nodes(data=True)
        ]
        relations = [
            {"source": u, "target": v, "type": a.get("type", "related_to"),
             "description": a.get("description", "")}
            for u, v, a in graph.edges(data=True)
        ]
        return {"entities": entities, "relations": relations}

    async def search_communities(
        self, query_vector: List[float], user_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        return _engine.search_communities(await self._load_communities(user_id), query_vector, top_k)

    async def communities_for_entities(self, seed_ids: List[str], user_id: str) -> List[Dict[str, Any]]:
        return _engine.communities_for_entities(await self._load_communities(user_id), seed_ids)

    # --- delete -------------------------------------------------------------
    async def delete_by_document(self, document_id: str, user_id: str) -> None:
        pool = await self._pool_or_create()
        prefix = f"{document_id}{_SEP}"
        statements = [
            ("MATCH ()-[r {document_id: $document_id}]->() DELETE r", {"document_id": document_id}),
            ("""MATCH (e:Entity {user_id: $user_id}) WHERE e.chunks IS NOT NULL
                SET e.chunks = [m IN e.chunks WHERE NOT m STARTS WITH $prefix]""",
             {"user_id": user_id, "prefix": prefix}),
            ("""MATCH (e:Entity {user_id: $user_id})
                WHERE e.chunks IS NULL OR size(e.chunks) = 0
                DETACH DELETE e""", {"user_id": user_id}),
        ]
        async with pool.acquire() as conn:
            for cypher, params in statements:
                await self._run_cypher(conn, cypher, params)
        logger.debug("Graph(AGE): deleted doc %s for user %s (relations + mentions + orphan entities)",
                     document_id, user_id)
