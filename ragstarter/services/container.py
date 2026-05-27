from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from ragstarter.core.config import Settings, get_settings
from ragstarter.db.models import DocumentRecord
from ragstarter.db.session import build_engine, build_session_factory, init_db, session_scope
from ragstarter.services.chunking import Chunker
from ragstarter.services.embeddings import EmbeddingService, build_embedding_service
from ragstarter.services.evaluator import Evaluator
from ragstarter.services.llm import build_provider
from ragstarter.services.rag import RagService
from ragstarter.services.retriever import Retriever
from ragstarter.services.text_loading import LoadedDocument, load_file, load_from_url
from ragstarter.services.vector_store import VectorStore


@dataclass(slots=True)
class Services:
    settings: Settings
    engine: object
    session_factory: sessionmaker
    embeddings: EmbeddingService
    store: VectorStore
    chunker: Chunker
    retriever: Retriever
    provider: object
    rag: RagService
    evaluator: Evaluator


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)
    embeddings = build_embedding_service(settings)
    store = VectorStore(settings)
    chunker = Chunker(settings.chunk_size, settings.chunk_overlap)
    retriever = Retriever(store, embeddings)
    provider = build_provider(settings)
    rag = RagService(retriever, provider, max_context_chunks=settings.max_context_chunks)
    evaluator = Evaluator()
    return Services(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        embeddings=embeddings,
        store=store,
        chunker=chunker,
        retriever=retriever,
        provider=provider,
        rag=rag,
        evaluator=evaluator,
    )


def ingest_file(services: Services, file_path: Path, source_name: str | None = None):
    doc = load_file(file_path)
    return ingest_loaded_document(services, doc)


def ingest_url(services: Services, url: str):
    doc = load_from_url(url)
    return ingest_loaded_document(services, doc)


def ingest_loaded_document(services: Services, doc: LoadedDocument):
    from ragstarter.db.models import DocumentRecord

    chunks = services.chunker.chunk_text(
        text=doc.text,
        document_id=doc.content_hash,
        source_name=doc.source_name,
        source_type=doc.source_type,
        title=doc.title,
    )

    with session_scope(services.session_factory) as session:
        existing = session.query(DocumentRecord).filter(DocumentRecord.content_hash == doc.content_hash).one_or_none()
        if existing:
            services.store.delete_document(existing.id)
            session.delete(existing)
            session.flush()

        # Use a stable id derived from content hash so replacing the same file
        # produces a clean overwrite.
        document_id = doc.content_hash[:36]
        record = DocumentRecord(
            id=document_id,
            source_name=doc.source_name,
            source_type=doc.source_type,
            source_uri=doc.source_uri,
            content_hash=doc.content_hash,
            title=doc.title,
            status="indexed",
            chunk_count=len(chunks),
            text_length=doc.text_length,
            error=None,
        )
        session.add(record)
        session.flush()

    # Rebuild chunk ids to reflect the stored document id.
    for idx, chunk in enumerate(chunks):
        chunk.chunk_id = f"{document_id}:{idx}"
        chunk.document_id = document_id
        chunk.metadata["document_id"] = document_id

    services.store.upsert_chunks(chunks, services.embeddings)
    return record, chunks


def list_documents(services: Services):
    with session_scope(services.session_factory) as session:
        docs = session.query(DocumentRecord).order_by(DocumentRecord.created_at.desc()).all()
        return [doc.to_dict() for doc in docs]


def delete_document(services: Services, document_id: str) -> bool:
    from ragstarter.db.models import DocumentRecord

    with session_scope(services.session_factory) as session:
        doc = session.get(DocumentRecord, document_id)
        if not doc:
            return False
        services.store.delete_document(document_id)
        session.delete(doc)
        return True


def reset_all(services: Services) -> None:
    from ragstarter.db.models import DocumentRecord

    services.store.reset()
    with session_scope(services.session_factory) as session:
        session.query(DocumentRecord).delete()
