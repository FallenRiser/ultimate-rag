"""GraphRAG build-time logic: cluster the tenant graph into communities and LLM-summarise them
(global/thematic retrieval), and collapse multi-source entity descriptions. The graph *store*
owns persistence + retrieval primitives; this service owns the LLM/clustering orchestration."""

import logging
from typing import Dict, List

import networkx as nx
from networkx.algorithms import community as nx_community

from app.models.ingestion import CommunitySummary
from app.prompts.ingestion import DESCRIPTION_SUMMARY_SYSTEM, community_summary_system
from app.repositories.base import BaseGraphStore
from app.services.embeddings.base import BaseEmbeddingProvider
from app.services.llm.base import BaseLLMProvider
from app.utils.config import get_settings
from app.utils.text import cap_text

logger = logging.getLogger(__name__)

_MULTI_SOURCE = " | "   # marker left by the store when an entity's description was merged


class GraphRAGService:
    def __init__(
        self,
        graph_store: BaseGraphStore,
        llm: BaseLLMProvider,
        embedding: BaseEmbeddingProvider,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embedding = embedding

    async def summarize_entity_descriptions(self, entity_ids: List[str], user_id: str) -> None:
        """Gap 5: collapse the descriptions of entities that gathered several (across documents)
        into one coherent description. One LLM call per multi-source entity; failures are skipped."""
        data = await self.graph_store.load_graph_data(user_id)
        by_id = {e["id"]: e for e in data["entities"]}

        updates: Dict[str, str] = {}
        for entity_id in set(entity_ids):
            entity = by_id.get(entity_id)
            if not entity or _MULTI_SOURCE not in (entity.get("description") or ""):
                continue
            try:
                merged = await self.llm.chat([
                    {"role": "system", "content": DESCRIPTION_SUMMARY_SYSTEM},
                    {"role": "user", "content": entity["description"]},
                ])
            except Exception as exc:
                logger.warning("Description summarisation failed for %s: %s", entity_id, exc)
                logger.debug("Description summary failure (entity=%s)", entity_id, exc_info=True)
                continue
            if merged.strip():
                updates[entity_id] = merged.strip()

        if updates:
            logger.info("Summarised %d merged entity descriptions for %s", len(updates), user_id)
            await self.graph_store.update_entity_descriptions(updates, user_id)

    async def rebuild_communities(self, user_id: str) -> int:
        """Gap 3: Louvain-cluster the tenant graph, LLM-summarise each community, and store the
        reports (embedded for global search). Re-summarises every qualifying community — expensive,
        so it runs only when enrichment.community_detection.enabled. Returns #communities written."""
        cfg = get_settings().enrichment.community_detection
        data = await self.graph_store.load_graph_data(user_id)
        if not data["entities"]:
            await self.graph_store.save_communities([], user_id)
            return 0

        graph = nx.Graph()
        graph.add_nodes_from(e["id"] for e in data["entities"])
        for relation in data["relations"]:
            graph.add_edge(relation["source"], relation["target"])

        clusters = nx_community.louvain_communities(graph, seed=42)
        by_id = {e["id"]: e for e in data["entities"]}

        communities: List[Dict] = []
        for index, members in enumerate(clusters):
            if len(members) < cfg.min_community_size:
                continue
            context = self._community_context(members, by_id, data["relations"], cfg.max_chars)
            try:
                report: CommunitySummary = await self.llm.structured_output(
                    messages=[
                        {"role": "system", "content": community_summary_system(cfg.max_chars)},
                        {"role": "user", "content": context},
                    ],
                    schema=CommunitySummary,
                )
            except Exception as exc:
                logger.warning("Community summary failed for cluster %d (%s): %s", index, user_id, exc)
                logger.debug("Community summary failure (cluster=%d)", index, exc_info=True)
                continue

            embedding = await self._embed(f"{report.title}\n{report.summary}")
            communities.append({
                "id": str(index),
                "title": report.title,
                "summary": report.summary,
                "members": sorted(members),
                "embedding": embedding,
            })

        await self.graph_store.save_communities(communities, user_id)
        logger.info("Built %d communities for %s (from %d clusters)", len(communities), user_id, len(clusters))
        return len(communities)

    @staticmethod
    def _community_context(members, by_id: dict, relations: List[dict], max_chars: int) -> str:
        member_set = set(members)
        lines = ["Entities:"]
        for entity_id in sorted(member_set):
            entity = by_id.get(entity_id, {})
            desc = entity.get("description", "")
            lines.append(f"- {entity.get('name', entity_id)} ({entity.get('type', 'Entity')}): {desc}")
        rels = [
            f"- {by_id.get(r['source'], {}).get('name', r['source'])} "
            f"--{r.get('type', 'related_to')}--> "
            f"{by_id.get(r['target'], {}).get('name', r['target'])}: {r.get('description', '')}"
            for r in relations if r["source"] in member_set and r["target"] in member_set
        ]
        if rels:
            lines.append("Relationships:")
            lines.extend(rels)
        return cap_text("\n".join(lines), max_chars)

    async def _embed(self, text: str) -> List[float]:
        # Non-breaking: communities still store/search by membership even if embedding fails.
        try:
            return await self.embedding.embed_query(text)
        except Exception as exc:
            logger.warning("Community embedding failed; storing without vector: %s", exc)
            logger.debug("Community embedding failure", exc_info=True)
            return []


if __name__ == "__main__":
    # Self-check: description merge-summarisation (gap 5) and community build (gap 3).
    import asyncio
    import tempfile

    class _FakeLLM(BaseLLMProvider):
        async def chat(self, messages, **kwargs) -> str:
            return "MERGED DESCRIPTION"

        async def structured_output(self, messages, schema, **kwargs):
            return CommunitySummary(title="Theme", summary="A short report.")

    class _FakeEmb(BaseEmbeddingProvider):
        async def embed_documents(self, texts):
            return [[float(len(t)), 1.0] for t in texts]

        async def embed_query(self, text):
            return [float(len(text)), 1.0]

        @property
        def dim(self) -> int:
            return 2

    async def _demo() -> None:
        from app.repositories.graph.networkx_store import NetworkXRepository

        get_settings().enrichment.community_detection.min_community_size = 2
        store = NetworkXRepository(graph_dir=tempfile.mkdtemp())
        await store.upsert_entities([
            {"id": "nvidia", "name": "Nvidia", "type": "Org", "description": "A | B", "document_id": "d", "chunk_id": "c1"},
            {"id": "cuda", "name": "CUDA", "type": "Product", "description": "plat", "document_id": "d", "chunk_id": "c2"},
            {"id": "python", "name": "Python", "type": "Product", "description": "lang", "document_id": "d", "chunk_id": "c3"},
            {"id": "guido", "name": "Guido", "type": "Person", "description": "author", "document_id": "d", "chunk_id": "c4"},
        ], "u")
        await store.upsert_relations([
            {"source_id": "nvidia", "target_id": "cuda", "relation_type": "makes", "document_id": "d"},
            {"source_id": "guido", "target_id": "python", "relation_type": "created", "document_id": "d"},
        ], "u")

        svc = GraphRAGService(store, _FakeLLM(), _FakeEmb())

        await svc.summarize_entity_descriptions(["nvidia", "cuda"], "u")
        by_id = {e["id"]: e for e in (await store.load_graph_data("u"))["entities"]}
        assert by_id["nvidia"]["description"] == "MERGED DESCRIPTION"   # multi-source → summarised
        assert by_id["cuda"]["description"] == "plat"                   # single-source → untouched

        count = await svc.rebuild_communities("u")
        assert count == 2                                               # two clusters → two reports
        assert (await store.communities_for_entities(["nvidia"], "u"))[0]["title"] == "Theme"
        assert len(await store.search_communities([4.0, 1.0], "u", 5)) == 2
        print("OK")

    asyncio.run(_demo())
