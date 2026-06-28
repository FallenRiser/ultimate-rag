from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseVectorDB(ABC):
    """Abstract base for vector stores. Concrete class sets supports_sparse to declare its capabilities."""

    supports_sparse: bool = False  # qdrant = True; pgvector = False

    @abstractmethod
    async def initialize_collection(self, vector_size: int) -> None: ...

    @abstractmethod
    async def insert(self, chunks: List[Dict[str, Any]], user_id: str) -> bool: ...

    @abstractmethod
    async def dense_search(
        self,
        query_vector: List[float],
        user_id: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def fetch_by_ids(
        self, chunk_ids: List[str], user_id: str
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def list_by_document(
        self, document_id: str, user_id: str
    ) -> List[Dict[str, Any]]: ...

    async def sparse_search(
        self,
        query_text: str,
        user_id: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support sparse/BM25 search. "
            "Switch vector_store.provider to 'qdrant' or use mode='semantic'."
        )

    @abstractmethod
    async def delete_by_document(self, document_id: str, user_id: str) -> bool: ...

    @abstractmethod
    async def delete_by_version(self, version_id: str, user_id: str) -> bool: ...


class BaseGraphStore(ABC):
    """Abstract base for graph stores (Apache AGE on Postgres / NetworkX local files).

    Entities merge across a tenant's documents by normalized name; each carries a description
    and (optionally) an embedding for semantic matching. Relations are edges. Communities are
    LLM-summarised clusters used for global/thematic retrieval."""

    # --- writes (ingest) ----------------------------------------------------
    @abstractmethod
    async def upsert_entities(self, entities: List[Dict[str, Any]], user_id: str) -> None:
        """Batch upsert. Each entity: {id, name, type, description, embedding, document_id, chunk_id}.
        Merges chunk mentions and descriptions onto the existing node (cross-document)."""
        ...

    @abstractmethod
    async def upsert_relations(self, relations: List[Dict[str, Any]], user_id: str) -> None:
        """Batch upsert. Each relation: {source_id, target_id, relation_type, description, document_id}."""
        ...

    @abstractmethod
    async def update_entity_descriptions(self, descriptions: Dict[str, str], user_id: str) -> None:
        """Overwrite the description of specific entities (used after LLM merge-summarisation)."""
        ...

    @abstractmethod
    async def save_communities(self, communities: List[Dict[str, Any]], user_id: str) -> None:
        """Replace the tenant's community reports. Each: {id, title, summary, members, embedding}."""
        ...

    # --- reads (retrieve / community build) ---------------------------------
    @abstractmethod
    async def search_entities(
        self, query_vector: List[float], user_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """Semantic seed selection: entities ranked by embedding cosine. [{id, name, score}]."""
        ...

    @abstractmethod
    async def match_entities_by_name(
        self, names: List[str], user_id: str
    ) -> List[Dict[str, Any]]:
        """String fallback seed selection (partial, case-insensitive). [{id, name, score}]."""
        ...

    @abstractmethod
    async def expand_and_collect(
        self, seed_ids: List[str], user_id: str, hops: int
    ) -> List[str]:
        """Traverse `hops` relation-edges from the seeds and return their chunk_ids, ranked
        seeds-first then by hop distance (hops=0 = seeds only)."""
        ...

    @abstractmethod
    async def load_graph_data(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Whole tenant graph for clustering/summarisation:
        {"entities": [{id, name, type, description}], "relations": [{source, target, type, description}]}."""
        ...

    @abstractmethod
    async def search_communities(
        self, query_vector: List[float], user_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """Global/thematic retrieval: community reports ranked by cosine. [{id, title, summary, score}]."""
        ...

    @abstractmethod
    async def communities_for_entities(
        self, seed_ids: List[str], user_id: str
    ) -> List[Dict[str, Any]]:
        """High-level dual-level signal: community reports containing any seed entity."""
        ...

    @abstractmethod
    async def delete_by_document(self, document_id: str, user_id: str) -> None: ...


class BaseCacheBackend(ABC):
    """Abstract base for query cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]: ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...
