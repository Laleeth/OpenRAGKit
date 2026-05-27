from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    id: str
    source_name: str
    source_type: str
    source_uri: str | None = None
    content_hash: str
    title: str | None = None
    status: str
    chunk_count: int
    text_length: int
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class UploadResponse(BaseModel):
    document: DocumentInfo
    message: str
    indexed_chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    source_type: str | None = None
    document_id: str | None = None


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    source_name: str
    source_type: str
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    chunk_index: int | None = None
    score: float
    content: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: list[SearchHit]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    source_type: str | None = None
    document_id: str | None = None


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_name: str
    source_type: str
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    chunk_index: int | None = None
    score: float
    content: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    provider: str
    sources: list[SourceChunk]
    latency_ms: int


class EvaluationRequest(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)


class EvaluationResponse(BaseModel):
    grounded_sentence_ratio: float
    citation_coverage: float
    unique_sources: int
    answer_length: int
    notes: list[str]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    documents_indexed: int
