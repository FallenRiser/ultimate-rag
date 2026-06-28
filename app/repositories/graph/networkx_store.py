# NetworkX — embedded, zero-infra graph store. One JSON file per user (a per-tenant graph),
# held as a DiGraph, plus a sidecar JSON of community reports. Pairs with chroma + sqlite for a
# fully local, no-Docker stack. Graph algorithms live in app.repositories.graph._engine.
#
# ponytail: entity embeddings are stored inline in the topology JSON. Fine to ~10k entities/user;
# beyond that move vectors to a sidecar .npz or switch graph_store.provider to age (pgvector).

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

from app.repositories.base import BaseGraphStore
from app.repositories.graph import _engine
from app.utils.config import get_settings

logger = logging.getLogger(__name__)

_SEP = _engine.SEP


class NetworkXRepository(BaseGraphStore):
    """Per-user graph at {graph_dir}/{user_id}.json and communities at {user_id}.communities.json.
    The per-user asyncio.Lock serialises read-modify-write within one process — for multi-worker
    deployments use AGE."""

    def __init__(self, graph_dir: str = "graph_data"):
        self._dir = Path(graph_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def _path(self, user_id: str) -> Path:
        # Path(...).name strips directory separators → blocks path traversal via user_id.
        return self._dir / f"{Path(user_id).name}.json"

    def _communities_path(self, user_id: str) -> Path:
        return self._dir / f"{Path(user_id).name}.communities.json"

    # --- persistence --------------------------------------------------------
    @staticmethod
    def _serialize(graph: nx.DiGraph) -> dict:
        return {
            "nodes": [{"id": n, **attrs} for n, attrs in graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **attrs} for u, v, attrs in graph.edges(data=True)],
        }

    @staticmethod
    def _deserialize(data: dict) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node in data.get("nodes", []):
            graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
        for edge in data.get("edges", []):
            attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
            graph.add_edge(edge["source"], edge["target"], **attrs)
        return graph

    def _load_sync(self, user_id: str) -> nx.DiGraph:
        path = self._path(user_id)
        if not path.exists():
            return nx.DiGraph()
        with open(path, encoding="utf-8") as f:
            return self._deserialize(json.load(f))

    def _save_sync(self, user_id: str, graph: nx.DiGraph) -> None:
        self._atomic_write(self._path(user_id), self._serialize(graph))

    def _load_communities_sync(self, user_id: str) -> List[Dict[str, Any]]:
        path = self._communities_path(user_id)
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _atomic_write(path: Path, data: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)   # atomic on POSIX and Windows

    async def _load(self, user_id: str) -> nx.DiGraph:
        return await asyncio.to_thread(self._load_sync, user_id)

    async def _save(self, user_id: str, graph: nx.DiGraph) -> None:
        await asyncio.to_thread(self._save_sync, user_id, graph)

    # --- writes -------------------------------------------------------------
    async def upsert_entities(self, entities: List[Dict[str, Any]], user_id: str) -> None:
        if not entities:
            return
        cap = get_settings().enrichment.entity_extraction.description_max_chars
        async with self._lock(user_id):
            graph = await self._load(user_id)
            for entity in entities:
                node_id = entity["id"]
                existing = graph.nodes.get(node_id, {})
                chunks = list(existing.get("chunks", []))
                chunk_id = entity.get("chunk_id", "")
                if chunk_id:
                    mention = f'{entity.get("document_id", "")}{_SEP}{chunk_id}'
                    if mention not in chunks:
                        chunks.append(mention)
                graph.add_node(
                    node_id,
                    name=entity.get("name", node_id),
                    type=entity.get("type", "Entity"),
                    description=_engine.merge_description(
                        existing.get("description", ""), entity.get("description", ""), cap
                    ),
                    embedding=entity.get("embedding") or existing.get("embedding"),
                    chunks=chunks,
                )
            await self._save(user_id, graph)
        logger.debug("Graph: upserted %d entity mention(s) for user %s (%d nodes total)",
                     len(entities), user_id, graph.number_of_nodes())

    async def upsert_relations(self, relations: List[Dict[str, Any]], user_id: str) -> None:
        if not relations:
            return
        async with self._lock(user_id):
            graph = await self._load(user_id)
            for relation in relations:
                if relation["source_id"] not in graph or relation["target_id"] not in graph:
                    continue   # only link entities that exist — no phantom nodes
                graph.add_edge(
                    relation["source_id"],
                    relation["target_id"],
                    type=relation.get("relation_type", "related_to"),
                    description=relation.get("description", ""),
                    document_id=relation.get("document_id", ""),
                )
            await self._save(user_id, graph)
        logger.debug("Graph: upserted %d relation(s) for user %s (%d edges total)",
                     len(relations), user_id, graph.number_of_edges())

    async def update_entity_descriptions(self, descriptions: Dict[str, str], user_id: str) -> None:
        if not descriptions:
            return
        async with self._lock(user_id):
            graph = await self._load(user_id)
            for node_id, text in descriptions.items():
                if node_id in graph:
                    graph.nodes[node_id]["description"] = text
            await self._save(user_id, graph)

    async def save_communities(self, communities: List[Dict[str, Any]], user_id: str) -> None:
        async with self._lock(user_id):
            await asyncio.to_thread(self._atomic_write, self._communities_path(user_id), communities)
        logger.debug("Graph: saved %d community report(s) for user %s", len(communities), user_id)

    # --- reads --------------------------------------------------------------
    async def search_entities(
        self, query_vector: List[float], user_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        graph = await self._load(user_id)
        results = _engine.search_entities(graph, query_vector, top_k)
        logger.debug("Graph: entity search for user %s -> %d hit(s)", user_id, len(results))
        return results

    async def match_entities_by_name(self, names: List[str], user_id: str) -> List[Dict[str, Any]]:
        graph = await self._load(user_id)
        results = _engine.match_entities_by_name(graph, names)
        logger.debug("Graph: name match %s for user %s -> %d hit(s)", names, user_id, len(results))
        return results

    async def expand_and_collect(self, seed_ids: List[str], user_id: str, hops: int) -> List[str]:
        if not seed_ids:
            return []
        graph = await self._load(user_id)
        chunk_ids = _engine.expand_and_collect(graph, seed_ids, hops)
        logger.debug("Graph: expand %d seed(s) (%d hops) for user %s -> %d chunk(s)",
                     len(seed_ids), hops, user_id, len(chunk_ids))
        return chunk_ids

    async def load_graph_data(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        graph = await self._load(user_id)
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
        communities = await asyncio.to_thread(self._load_communities_sync, user_id)
        return _engine.search_communities(communities, query_vector, top_k)

    async def communities_for_entities(self, seed_ids: List[str], user_id: str) -> List[Dict[str, Any]]:
        communities = await asyncio.to_thread(self._load_communities_sync, user_id)
        return _engine.communities_for_entities(communities, seed_ids)

    # --- delete -------------------------------------------------------------
    async def delete_by_document(self, document_id: str, user_id: str) -> None:
        async with self._lock(user_id):
            graph = await self._load(user_id)
            prefix = f"{document_id}{_SEP}"
            changed = False

            orphans: List[str] = []
            for node, attrs in graph.nodes(data=True):
                chunks = attrs.get("chunks", [])
                kept = [m for m in chunks if not m.startswith(prefix)]
                if len(kept) != len(chunks):
                    attrs["chunks"] = kept
                    changed = True
                if not kept:
                    orphans.append(node)

            doc_edges = [(u, v) for u, v, a in graph.edges(data=True) if a.get("document_id") == document_id]
            if doc_edges:
                graph.remove_edges_from(doc_edges)
                changed = True
            if orphans:
                graph.remove_nodes_from(orphans)   # also drops their incident edges
                changed = True
            if changed:
                await self._save(user_id, graph)
        logger.debug("Graph: deleted doc %s for user %s (removed %d orphan node(s), %d edge(s))",
                     document_id, user_id, len(orphans), len(doc_edges))


if __name__ == "__main__":
    import tempfile

    async def _demo() -> None:
        repo = NetworkXRepository(graph_dir=tempfile.mkdtemp())
        # d1: Nvidia (c1) -makes-> CUDA (c2); d2 re-mentions Nvidia (c3) → must merge to one node.
        await repo.upsert_entities([
            {"id": "nvidia", "name": "Nvidia", "type": "Org", "description": "A chipmaker.",
             "embedding": [1.0, 0.0], "document_id": "d1", "chunk_id": "c1"},
            {"id": "cuda", "name": "CUDA", "description": "A platform.",
             "embedding": [0.0, 1.0], "document_id": "d1", "chunk_id": "c2"},
        ], "alice")
        await repo.upsert_relations([
            {"source_id": "nvidia", "target_id": "cuda", "relation_type": "makes", "document_id": "d1"},
        ], "alice")
        await repo.upsert_entities([
            {"id": "nvidia", "name": "Nvidia", "type": "Org", "description": "Designs GPUs.",
             "embedding": [0.9, 0.1], "document_id": "d2", "chunk_id": "c3"},
        ], "alice")

        assert await repo.search_entities([1.0, 0.0], "bob", 5) == []                     # tenant isolation
        top = await repo.search_entities([1.0, 0.0], "alice", 1)                          # semantic match
        assert top and top[0]["id"] == "nvidia"
        data = await repo.load_graph_data("alice")
        nvidia = next(e for e in data["entities"] if e["id"] == "nvidia")
        assert "chipmaker" in nvidia["description"] and "GPUs" in nvidia["description"]    # descriptions merged

        seeds = [s["id"] for s in await repo.search_entities([1.0, 0.0], "alice", 1)]
        assert sorted(await repo.expand_and_collect(seeds, "alice", 0)) == ["c1", "c3"]    # seeds only (merged)
        assert sorted(await repo.expand_and_collect(seeds, "alice", 1)) == ["c1", "c2", "c3"]  # +1 hop → CUDA

        await repo.save_communities([
            {"id": "0", "title": "GPU stack", "summary": "Nvidia and CUDA.",
             "members": ["nvidia", "cuda"], "embedding": [1.0, 0.0]},
        ], "alice")
        assert (await repo.communities_for_entities(["nvidia"], "alice"))[0]["title"] == "GPU stack"
        assert (await repo.search_communities([1.0, 0.0], "alice", 5))[0]["id"] == "0"

        await repo.delete_by_document("d1", "alice")
        assert await repo.expand_and_collect(["nvidia"], "alice", 1) == ["c3"]             # d1 gone, node survives
        assert await repo.expand_and_collect(["cuda"], "alice", 1) == []                   # CUDA orphaned → removed
        print("OK")

    asyncio.run(_demo())
