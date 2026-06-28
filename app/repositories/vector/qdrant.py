import asyncio
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    SparseVector,
    VectorParams,
)

from app.repositories.base import BaseVectorDB


class QdrantRepository(BaseVectorDB):
    supports_sparse = True  # qdrant: semantic + BM25

    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        collection_name: str = "rag_chunks",
        sparse_model: str = "Qdrant/bm25",
    ):
        self.client = AsyncQdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.sparse_model_name = sparse_model
        self._sparse_model = None  # lazily loaded fastembed model (shared by insert + search)

    def _get_sparse_model(self):
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding
            self._sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name)
        return self._sparse_model

    def _build_filter(self, user_id: str, filters: Optional[Dict[str, Any]] = None) -> Filter:
        # user_id is always enforced. A scalar filter matches exactly; a list matches ANY of
        # its values (OR within the key). Different keys AND together.
        must = [FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))]
        for key, value in (filters or {}).items():
            if isinstance(value, (list, tuple, set)):
                match = MatchAny(any=list(value))
            else:
                match = MatchValue(value=value)
            must.append(FieldCondition(key=f"metadata.{key}", match=match))
        return Filter(must=must)

    async def _embed_sparse(self, texts: List[str]) -> List[SparseVector]:
        # fastembed is synchronous — run in a thread so we don't block the event loop
        model = self._get_sparse_model()
        embeddings = await asyncio.to_thread(lambda: list(model.embed(texts)))
        return [
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in embeddings
        ]

    async def initialize_collection(self, vector_size: int) -> None:
        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                sparse_vectors_config={"sparse": {}},
            )

    async def insert(self, chunks: List[Dict[str, Any]], user_id: str) -> bool:
        # Compute BM25 sparse vectors here — the capability-aware store owns sparse,
        # so the ingestion worker stays vector-store-agnostic.
        sparse_vectors = await self._embed_sparse([c["text"] for c in chunks])
        points = []
        for chunk, sparse in zip(chunks, sparse_vectors):
            chunk["metadata"]["user_id"] = user_id
            points.append(
                PointStruct(
                    id=chunk["chunk_id"],
                    vector={
                        "": chunk["dense_vector"],
                        "sparse": sparse,
                    },
                    payload={"text": chunk["text"], "metadata": chunk["metadata"]},
                )
            )
        await self.client.upsert(collection_name=self.collection_name, points=points)
        return True

    async def dense_search(
        self,
        query_vector: List[float],
        user_id: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._build_filter(user_id, filters),
            limit=top_k,
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results.points]

    async def sparse_search(
        self,
        query_text: str,
        user_id: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        vec = (await self._embed_sparse([query_text]))[0]
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=vec,
            using="sparse",
            query_filter=self._build_filter(user_id, filters),
            limit=top_k,
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results.points]

    async def fetch_by_ids(
        self, chunk_ids: List[str], user_id: str
    ) -> List[Dict[str, Any]]:
        if not chunk_ids:
            return []
        records = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=list(chunk_ids),
            with_payload=True,
        )
        out = []
        for rec in records:
            payload = rec.payload or {}
            # Enforce tenant isolation — never return another user's chunk
            if payload.get("metadata", {}).get("user_id") != user_id:
                continue
            out.append({"id": rec.id, "score": 1.0, "payload": payload})
        return out

    async def list_by_document(
        self, document_id: str, user_id: str
    ) -> List[Dict[str, Any]]:
        doc_filter = Filter(
            must=[
                FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id)),
            ]
        )
        out = []
        offset = None
        while True:
            points, offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=doc_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            out.extend({"id": p.id, "payload": p.payload} for p in points)
            if offset is None:
                break
        return out

    async def delete_by_document(self, document_id: str, user_id: str) -> bool:
        doc_filter = Filter(
            must=[
                FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id)),
            ]
        )
        await self.client.delete(collection_name=self.collection_name, points_selector=doc_filter)
        return True

    async def delete_by_version(self, version_id: str, user_id: str) -> bool:
        version_filter = Filter(
            must=[
                FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="metadata.version_id", match=MatchValue(value=version_id)),
            ]
        )
        await self.client.delete(
            collection_name=self.collection_name, points_selector=version_filter
        )
        return True
