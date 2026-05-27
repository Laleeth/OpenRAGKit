from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection

from ragstarter.core.config import Settings
from ragstarter.services.chunking import Chunk
from ragstarter.services.embeddings import EmbeddingService


@dataclass(slots=True)
class VectorHit:
    chunk_id: str
    document_id: str
    source_name: str
    source_type: str
    title: str | None
    page_start: int | None
    page_end: int | None
    chunk_index: int | None
    score: float
    content: str
    metadata: dict[str, object]


class VectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> Collection:
        return self._collection

    def upsert_chunks(self, chunks: list[Chunk], embedding_service: EmbeddingService) -> None:
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        embeddings = embedding_service.embed_texts(documents)
        metadatas = [chunk.metadata for chunk in chunks]
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        embedding_service: EmbeddingService,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[VectorHit]:
        query_embedding = embedding_service.embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[VectorHit] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for idx, chunk_id in enumerate(ids):
            metadata = metas[idx] or {}
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            score = max(0.0, 1.0 - distance)
            hits.append(
                VectorHit(
                    chunk_id=str(chunk_id),
                    document_id=str(metadata.get("document_id", "")),
                    source_name=str(metadata.get("source_name", "")),
                    source_type=str(metadata.get("source_type", "")),
                    title=metadata.get("title"),
                    page_start=metadata.get("page_start"),
                    page_end=metadata.get("page_end"),
                    chunk_index=metadata.get("chunk_index"),
                    score=score,
                    content=str(docs[idx]) if idx < len(docs) else "",
                    metadata=dict(metadata),
                )
            )
        return hits

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def reset(self) -> None:
        try:
            self._client.delete_collection("documents")
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:
            return 0
