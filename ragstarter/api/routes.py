from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from ragstarter.api.deps import get_services
from ragstarter.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)
from ragstarter.services.container import delete_document, ingest_file, ingest_loaded_document, ingest_url, list_documents, reset_all
from ragstarter.services.text_loading import load_file


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    services = get_services()
    return HealthResponse(
        status="ok",
        app_name=services.settings.app_name,
        environment=services.settings.environment,
        documents_indexed=services.store.count(),
    )


@router.get("/documents", response_model=list[DocumentInfo])
def documents():
    services = get_services()
    return [DocumentInfo.model_validate(doc) for doc in list_documents(services)]


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    services = get_services()
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    destination = services.settings.uploads_dir / f"{uuid4().hex}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)

    contents = await file.read()
    destination.write_bytes(contents)

    try:
        loaded = load_file(destination)
        record, chunks = ingest_loaded_document(services, loaded)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not ingest document: {exc}") from exc

    return UploadResponse(
        document=DocumentInfo.model_validate(record.to_dict()),
        message="Document indexed successfully.",
        indexed_chunks=len(chunks),
    )


@router.post("/documents/from-url", response_model=UploadResponse)
def upload_from_url(url: str = Form(...)):
    services = get_services()
    try:
        record, chunks = ingest_url(services, url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not ingest URL: {exc}") from exc

    return UploadResponse(
        document=DocumentInfo.model_validate(record.to_dict()),
        message="URL indexed successfully.",
        indexed_chunks=len(chunks),
    )


@router.delete("/documents/{document_id}")
def remove_document(document_id: str):
    services = get_services()
    if not delete_document(services, document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "document_id": document_id}


@router.post("/reset")
def reset_index():
    services = get_services()
    reset_all(services)
    return {"status": "reset"}


@router.post("/search", response_model=SearchResponse)
def search_documents(payload: SearchRequest):
    services = get_services()
    top_k = payload.top_k or services.settings.top_k
    hits = services.retriever.search(
        payload.query,
        top_k=top_k,
        source_type=payload.source_type,
        document_id=payload.document_id,
    )
    from ragstarter.schemas import SearchHit

    return SearchResponse(
        query=payload.query,
        top_k=top_k,
        hits=[
            SearchHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                source_name=hit.source_name,
                source_type=hit.source_type,
                title=hit.title,
                page_start=hit.page_start,
                page_end=hit.page_end,
                chunk_index=hit.chunk_index,
                score=round(hit.score, 4),
                content=hit.content,
            )
            for hit in hits
        ],
    )


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    services = get_services()
    top_k = payload.top_k or services.settings.top_k
    result = services.rag.answer(
        payload.question,
        top_k=top_k,
        source_type=payload.source_type,
        document_id=payload.document_id,
    )

    from ragstarter.schemas import SourceChunk

    return ChatResponse(
        question=result.question,
        answer=result.answer,
        provider=result.provider,
        latency_ms=result.latency_ms,
        sources=[
            SourceChunk(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                source_name=hit.source_name,
                source_type=hit.source_type,
                title=hit.title,
                page_start=hit.page_start,
                page_end=hit.page_end,
                chunk_index=hit.chunk_index,
                score=round(hit.score, 4),
                content=hit.content,
            )
            for hit in result.sources
        ],
    )


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest):
    services = get_services()
    return services.evaluator.evaluate(payload.question, payload.answer, payload.sources)


@router.get("/stats")
def stats():
    services = get_services()
    return {
        "documents": len(list_documents(services)),
        "chunks": services.store.count(),
        "provider": services.provider.name,
        "embedding_model": services.settings.embedding_model,
    }
