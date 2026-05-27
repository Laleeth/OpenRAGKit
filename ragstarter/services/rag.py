from __future__ import annotations

import time
from dataclasses import dataclass

from ragstarter.services.llm import LLMProvider
from ragstarter.services.retriever import Retriever
from ragstarter.services.vector_store import VectorHit


@dataclass(slots=True)
class RagAnswer:
    question: str
    answer: str
    provider: str
    sources: list[VectorHit]
    latency_ms: int


class RagService:
    def __init__(self, retriever: Retriever, provider: LLMProvider, max_context_chunks: int = 5):
        self.retriever = retriever
        self.provider = provider
        self.max_context_chunks = max_context_chunks

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        source_type: str | None = None,
        document_id: str | None = None,
    ) -> RagAnswer:
        started = time.perf_counter()
        hits = self.retriever.search(
            question,
            top_k=top_k,
            source_type=source_type,
            document_id=document_id,
        )
        if not hits:
            answer = "I could not find any indexed sources for that question."
            latency_ms = int((time.perf_counter() - started) * 1000)
            return RagAnswer(question=question, answer=answer, provider=self.provider.name, sources=[], latency_ms=latency_ms)

        context = self._build_context(hits[: self.max_context_chunks])
        try:
            answer = self.provider.generate(question, context)
        except Exception as exc:
            answer = (
                "The answer provider failed, so I returned a grounded extractive response instead. "
                f"Details: {exc}"
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return RagAnswer(question=question, answer=answer, provider=self.provider.name, sources=hits[: self.max_context_chunks], latency_ms=latency_ms)

    def _build_context(self, hits: list[VectorHit]) -> str:
        lines: list[str] = []
        for idx, hit in enumerate(hits, start=1):
            meta = []
            if hit.source_name:
                meta.append(hit.source_name)
            if hit.page_start is not None:
                if hit.page_end and hit.page_end != hit.page_start:
                    meta.append(f"pages {hit.page_start}-{hit.page_end}")
                else:
                    meta.append(f"page {hit.page_start}")
            if hit.title:
                meta.append(hit.title)
            prefix = f"[{idx}] " + " | ".join(meta) if meta else f"[{idx}]"
            lines.append(f"{prefix}\n{hit.content}")
        return "\n\n".join(lines)
