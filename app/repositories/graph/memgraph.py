# Memgraph — Cypher graph store over the Bolt protocol (via the neo4j async driver).
# Apache-friendly, single container, no storage backend to configure.

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from app.repositories.base import BaseGraphStore

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


class MemgraphRepository(BaseGraphStore):
    def __init__(self, url: str, user: str = "", password: str = ""):
        self._url = url
        self._auth = (user, password) if user else None
        self._driver = None  # lazily created (Bolt connection)

    def _get_driver(self) -> "AsyncDriver":
        if self._driver is None:
            from neo4j import AsyncGraphDatabase
            self._driver = AsyncGraphDatabase.driver(self._url, auth=self._auth)
        return self._driver

    async def upsert_entity(self, entity: Dict[str, Any], user_id: str) -> None:
        cypher = """
            MERGE (e:Entity {id: $id, user_id: $user_id})
            SET e.name = $name,
                e.type = $type,
                e.document_id = $document_id,
                e.chunk_id = $chunk_id
        """
        params = {
            "id": entity["id"],
            "user_id": user_id,
            "name": entity.get("name", entity["id"]),
            "type": entity.get("type", "Entity"),
            "document_id": entity.get("document_id", ""),
            "chunk_id": entity.get("chunk_id", ""),
        }
        async with self._get_driver().session() as session:
            await session.run(cypher, params)

    async def upsert_relation(self, relation: Dict[str, Any], user_id: str) -> None:
        # Relationship type can't be parameterised in Cypher, so it's stored as a property
        # on a fixed RELATED edge — keeps the query injection-free.
        cypher = """
            MATCH (a {id: $source_id, user_id: $user_id})
            MATCH (b {id: $target_id, user_id: $user_id})
            MERGE (a)-[r:RELATED {type: $type, document_id: $document_id}]->(b)
        """
        params = {
            "source_id": relation["source_id"],
            "target_id": relation["target_id"],
            "user_id": user_id,
            "type": relation.get("relation_type", "RELATED_TO"),
            "document_id": relation.get("document_id", ""),
        }
        async with self._get_driver().session() as session:
            await session.run(cypher, params)

    async def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        async with self._get_driver().session() as session:
            result = await session.run(cypher, params or {})
            return [record.data() async for record in result]

    async def find_chunks_for_entities(self, entity_names: List[str], user_id: str) -> List[str]:
        if not entity_names:
            return []
        # Case-insensitive partial match: a term matches if it appears in the entity name,
        # or the name appears in the term (handles "nvidia" ~ "Nvidia Corporation").
        cypher = """
            MATCH (e {user_id: $user_id})
            WHERE e.chunk_id <> ''
              AND any(t IN $names WHERE toLower(e.name) CONTAINS t OR t CONTAINS toLower(e.name))
            RETURN DISTINCT e.chunk_id AS chunk_id
        """
        async with self._get_driver().session() as session:
            result = await session.run(cypher, {"user_id": user_id, "names": entity_names})
            return [record["chunk_id"] async for record in result if record["chunk_id"]]

    async def delete_by_document(self, document_id: str, user_id: str) -> None:
        cypher = "MATCH (e {document_id: $document_id, user_id: $user_id}) DETACH DELETE e"
        async with self._get_driver().session() as session:
            await session.run(cypher, {"document_id": document_id, "user_id": user_id})
