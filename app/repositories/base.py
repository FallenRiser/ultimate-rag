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
    """Abstract base for graph stores (Apache AGE / Memgraph)."""

    @abstractmethod
    async def upsert_entity(self, entity: Dict[str, Any], user_id: str) -> None: ...

    @abstractmethod
    async def upsert_relation(self, relation: Dict[str, Any], user_id: str) -> None: ...

    @abstractmethod
    async def query(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def find_chunks_for_entities(
        self, entity_names: List[str], user_id: str
    ) -> List[str]: ...

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
