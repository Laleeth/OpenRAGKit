from __future__ import annotations

from ragstarter.services.embeddings import EmbeddingService
from ragstarter.services.vector_store import VectorHit, VectorStore


class Retriever:
    def __init__(self, store: VectorStore, embeddings: EmbeddingService):
        self.store = store
        self.embeddings = embeddings

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        source_type: str | None = None,
        document_id: str | None = None,
    ) -> list[VectorHit]:
        where: dict | None = None
        clauses: list[dict] = []
        if source_type:
            clauses.append({"source_type": source_type})
        if document_id:
            clauses.append({"document_id": document_id})
        if clauses:
            where = {"$and": clauses} if len(clauses) > 1 else clauses[0]
        return self.store.search(query=query, embedding_service=self.embeddings, top_k=top_k, where=where)
