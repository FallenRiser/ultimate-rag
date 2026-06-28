import asyncio
from typing import Any, Dict, List, Optional

from app.repositories.base import BaseVectorDB


def _build_where(user_id: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Chroma where-clause: user_id always enforced, plus metadata matches. A scalar matches
    exactly; a list matches ANY of its values (OR within the key, via $in). Multiple
    conditions combine with $and."""
    conditions: List[Dict[str, Any]] = [{"user_id": user_id}]
    for key, value in (filters or {}).items():
        if isinstance(value, (list, tuple, set)):
            conditions.append({key: {"$in": list(value)}})
        else:
            conditions.append({key: value})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class ChromaRepository(BaseVectorDB):
    supports_sparse = False  # chroma: semantic only — no BM25

    def __init__(self, path: str, collection_name: str = "rag_chunks"):
        self.path = path
        self.collection_name = collection_name
        self._collection = None  # lazily created (chromadb client is synchronous)

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=self.path)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def initialize_collection(self, vector_size: int) -> None:
        # Chroma infers dimensionality from the first insert — just ensure it exists.
        await asyncio.to_thread(self._get_collection)

    async def insert(self, chunks: List[Dict[str, Any]], user_id: str) -> bool:
        ids, embeddings, documents, metadatas = [], [], [], []
        for chunk in chunks:
            chunk["metadata"]["user_id"] = user_id
            ids.append(chunk["chunk_id"])
            embeddings.append(chunk["dense_vector"])
            documents.append(chunk["text"])
            metadatas.append(_flatten(chunk["metadata"]))

        collection = self._get_collection()
        await asyncio.to_thread(
            lambda: collection.upsert(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )
        )
        return True

    async def dense_search(
        self,
        query_vector: List[float],
        user_id: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        collection = self._get_collection()
        where = _build_where(user_id, filters)
        result = await asyncio.to_thread(
            lambda: collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            {
                "id": ids[i],
                "score": 1.0 - distances[i],  # cosine distance → similarity
                "payload": {"text": documents[i], "metadata": metadatas[i]},
            }
            for i in range(len(ids))
        ]

    async def fetch_by_ids(
        self, chunk_ids: List[str], user_id: str
    ) -> List[Dict[str, Any]]:
        if not chunk_ids:
            return []
        collection = self._get_collection()
        result = await asyncio.to_thread(
            lambda: collection.get(
                ids=list(chunk_ids),
                where={"user_id": user_id},  # enforce tenant isolation
                include=["documents", "metadatas"],
            )
        )
        ids = result["ids"]
        documents = result["documents"]
        metadatas = result["metadatas"]
        return [
            {
                "id": ids[i],
                "score": 1.0,
                "payload": {"text": documents[i], "metadata": metadatas[i]},
            }
            for i in range(len(ids))
        ]

    async def list_by_document(
        self, document_id: str, user_id: str
    ) -> List[Dict[str, Any]]:
        collection = self._get_collection()
        result = await asyncio.to_thread(
            lambda: collection.get(
                where={"$and": [{"user_id": user_id}, {"document_id": document_id}]},
                include=["documents", "metadatas"],
            )
        )
        ids = result["ids"]
        documents = result["documents"]
        metadatas = result["metadatas"]
        return [
            {"id": ids[i], "payload": {"text": documents[i], "metadata": metadatas[i]}}
            for i in range(len(ids))
        ]

    async def delete_by_document(self, document_id: str, user_id: str) -> bool:
        collection = self._get_collection()
        await asyncio.to_thread(
            lambda: collection.delete(
                where={"$and": [{"user_id": user_id}, {"document_id": document_id}]}
            )
        )
        return True

    async def delete_by_version(self, version_id: str, user_id: str) -> bool:
        collection = self._get_collection()
        await asyncio.to_thread(
            lambda: collection.delete(
                where={"$and": [{"user_id": user_id}, {"version_id": version_id}]}
            )
        )
        return True


def _flatten(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma only accepts scalar metadata values — drop None and stringify the rest."""
    flat = {}
    for key, value in metadata.items():
        if value is None:
            continue
        flat[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return flat
