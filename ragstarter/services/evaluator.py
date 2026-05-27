from __future__ import annotations

import re
from dataclasses import dataclass

from ragstarter.schemas import EvaluationResponse, SourceChunk


class Evaluator:
    def evaluate(self, question: str, answer: str, sources: list[SourceChunk]) -> EvaluationResponse:
        answer_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        if not answer_sentences:
            answer_sentences = [answer.strip()] if answer.strip() else []

        source_text = " ".join(source.content for source in sources).lower()
        source_labels = {f"[{i}]" for i in range(1, len(sources) + 1)}

        grounded = 0
        for sentence in answer_sentences:
            tokens = self._tokens(sentence)
            if not tokens:
                continue
            overlap = len(tokens & set(self._tokens(source_text)))
            if overlap >= max(2, len(tokens) // 5):
                grounded += 1

        grounded_ratio = grounded / max(1, len(answer_sentences))
        citation_count = sum(1 for token in source_labels if token in answer)
        citation_coverage = min(1.0, citation_count / max(1, len(source_labels)))
        unique_sources = len({source.document_id for source in sources if source.document_id})

        notes: list[str] = []
        if citation_coverage < 0.5:
            notes.append("Answer has weak citation coverage.")
        if grounded_ratio < 0.5:
            notes.append("Answer sentences do not look strongly supported by the retrieved context.")
        if unique_sources == 0:
            notes.append("No source documents were attached.")

        return EvaluationResponse(
            grounded_sentence_ratio=round(grounded_ratio, 3),
            citation_coverage=round(citation_coverage, 3),
            unique_sources=unique_sources,
            answer_length=len(answer.split()),
            notes=notes,
        )

    def _tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
