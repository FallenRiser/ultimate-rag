# Pure graph-algorithm helpers shared by both graph stores. No I/O — the stores own
# persistence (NetworkX = local files, AGE = Postgres) and call these on an in-memory DiGraph.

from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

SEP = "\x1f"  # a mention is stored on an entity node as "{document_id}\x1f{chunk_id}"


def chunk_id_of(mention: str) -> str:
    return mention.split(SEP, 1)[1] if SEP in mention else mention


def build_graph(
    entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]
) -> nx.DiGraph:
    """Assemble a DiGraph from row dicts (used by stores that load topology from a DB)."""
    graph = nx.DiGraph()
    for entity in entities:
        graph.add_node(
            entity["id"],
            name=entity.get("name", entity["id"]),
            type=entity.get("type", "Entity"),
            description=entity.get("description") or "",
            chunks=entity.get("chunks") or [],
            embedding=entity.get("embedding"),
        )
    for relation in relations:
        if relation.get("source") in graph and relation.get("target") in graph:
            graph.add_edge(
                relation["source"], relation["target"],
                type=relation.get("type", "related_to"),
                description=relation.get("description") or "",
                document_id=relation.get("document_id", ""),
            )
    return graph


def merge_description(old: str, new: str, max_chars: int) -> str:
    """Accumulate distinct descriptions for an entity seen across chunks/documents, capped."""
    new = (new or "").strip()
    if not new or new in (old or ""):
        return old or ""
    merged = f"{old} | {new}".strip(" |") if old else new
    return merged[:max_chars] if max_chars > 0 else merged


def _matrix(vectors: List[List[float]]) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def _cosine_rank(
    query_vector: List[float], items: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    """items: [{..., "embedding": [...]}] → same dicts (minus embedding) with a "score", ranked."""
    vectors = [it["embedding"] for it in items if it.get("embedding")]
    rows = [it for it in items if it.get("embedding")]
    if not rows:
        return []
    q = _matrix([list(query_vector)])[0]
    scores = _matrix(vectors) @ q
    order = np.argsort(-scores)[:top_k]
    ranked = []
    for i in order:
        item = {k: v for k, v in rows[int(i)].items() if k != "embedding"}
        item["score"] = float(scores[int(i)])
        ranked.append(item)
    return ranked


def search_entities(graph: nx.DiGraph, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
    """Semantic seed selection: rank entity nodes by cosine of their embedding to the query."""
    items = [
        {"id": node, "name": attrs.get("name", node), "embedding": attrs.get("embedding")}
        for node, attrs in graph.nodes(data=True)
    ]
    return _cosine_rank(query_vector, items, top_k)


def match_entities_by_name(graph: nx.DiGraph, names: List[str]) -> List[Dict[str, Any]]:
    """String fallback seed selection: partial, case-insensitive, both directions."""
    terms = [n.lower() for n in names if n]
    if not terms:
        return []
    out = []
    for node, attrs in graph.nodes(data=True):
        name = (attrs.get("name") or "").lower()
        if name and any(t in name or name in t for t in terms):
            out.append({"id": node, "name": attrs.get("name", node), "score": 1.0})
    return out


def expand_and_collect(graph: nx.DiGraph, seed_ids: List[str], hops: int) -> List[str]:
    """BFS `hops` undirected edges out from seeds, then collect chunk_ids in traversal order
    (seeds first, then nearer hops) — the natural relevance ranking for local GraphRAG."""
    ordered: List[str] = []
    seen_nodes = set()
    frontier: List[str] = []
    for sid in seed_ids:
        if sid in graph and sid not in seen_nodes:
            seen_nodes.add(sid)
            ordered.append(sid)
            frontier.append(sid)

    for _ in range(max(0, hops)):
        nxt: List[str] = []
        for node in frontier:
            for neighbor in (*graph.successors(node), *graph.predecessors(node)):
                if neighbor not in seen_nodes:
                    seen_nodes.add(neighbor)
                    ordered.append(neighbor)
                    nxt.append(neighbor)
        frontier = nxt
        if not frontier:
            break

    seen_chunks = set()
    chunk_ids: List[str] = []
    for node in ordered:
        for mention in graph.nodes[node].get("chunks", []):
            cid = chunk_id_of(mention)
            if cid and cid not in seen_chunks:
                seen_chunks.add(cid)
                chunk_ids.append(cid)
    return chunk_ids


def communities_for_entities(
    communities: List[Dict[str, Any]], seed_ids: List[str]
) -> List[Dict[str, Any]]:
    """High-level (dual-level) signal: the community reports whose members include a seed entity."""
    seeds = set(seed_ids)
    out = []
    for community in communities:
        if seeds.intersection(community.get("members", [])):
            out.append({k: community.get(k) for k in ("id", "title", "summary")})
    return out


def search_communities(
    communities: List[Dict[str, Any]], query_vector: Optional[List[float]], top_k: int
) -> List[Dict[str, Any]]:
    """Global/thematic retrieval: rank community reports by cosine to the query (or first N
    if reports were never embedded)."""
    if query_vector and any(c.get("embedding") for c in communities):
        ranked = _cosine_rank(query_vector, [dict(c) for c in communities], top_k)
        by_id = {c["id"]: c for c in communities}
        return [{**{k: by_id[r["id"]].get(k) for k in ("id", "title", "summary")}, "score": r["score"]}
                for r in ranked]
    return [{k: c.get(k) for k in ("id", "title", "summary")} | {"score": 0.0}
            for c in communities[:top_k]]
