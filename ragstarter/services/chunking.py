from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    source_name: str
    source_type: str
    title: str | None
    page_start: int | None
    page_end: int | None
    metadata: dict[str, object]


class Chunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        *,
        text: str,
        document_id: str,
        source_name: str,
        source_type: str,
        title: str | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        blocks = self._split_into_blocks(text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for block in blocks:
            block_len = self._word_len(block)
            if block_len > self.chunk_size:
                if current:
                    chunks.append("\n\n".join(current).strip())
                    current, current_len = [], 0
                chunks.extend(self._split_large_block(block))
                continue

            proposed_len = current_len + block_len
            if current and proposed_len > self.chunk_size:
                chunks.append("\n\n".join(current).strip())
                current = self._apply_overlap(chunks[-1])
                current_len = self._word_len(" ".join(current))
            current.append(block)
            current_len = self._word_len("\n\n".join(current))

        if current:
            chunks.append("\n\n".join(current).strip())

        deduped: list[str] = []
        for chunk in chunks:
            cleaned = chunk.strip()
            if not cleaned:
                continue
            if deduped and cleaned == deduped[-1]:
                continue
            deduped.append(cleaned)

        out: list[Chunk] = []
        for idx, content in enumerate(deduped):
            page_start, page_end = self._extract_page_range(content)
            out.append(
                Chunk(
                    chunk_id=f"{document_id}:{idx}",
                    document_id=document_id,
                    content=content,
                    chunk_index=idx,
                    source_name=source_name,
                    source_type=source_type,
                    title=title,
                    page_start=page_start,
                    page_end=page_end,
                    metadata={
                        "document_id": document_id,
                        "source_name": source_name,
                        "source_type": source_type,
                        "title": title,
                        "chunk_index": idx,
                        "page_start": page_start,
                        "page_end": page_end,
                    },
                )
            )
        return out

    def _split_into_blocks(self, text: str) -> list[str]:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        blocks: list[str] = []
        current: list[str] = []

        for line in lines:
            if not line.strip():
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
                continue

            heading = _HEADING_RE.match(line)
            if heading:
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
                blocks.append(line.strip())
                continue

            current.append(line.rstrip())

        if current:
            blocks.append("\n".join(current).strip())

        merged = [b for b in blocks if b]
        return merged

    def _split_large_block(self, block: str) -> list[str]:
        sentences = _SENTENCE_SPLIT_RE.split(block)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sent_len = self._word_len(sentence)
            if current and current_len + sent_len > self.chunk_size:
                chunks.append(" ".join(current).strip())
                current = self._apply_overlap(chunks[-1])
                current_len = self._word_len(" ".join(current))
            current.append(sentence)
            current_len = self._word_len(" ".join(current))

        if current:
            chunks.append(" ".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _apply_overlap(self, previous_chunk: str) -> list[str]:
        words = previous_chunk.split()
        if not words:
            return []
        overlap = words[-self.chunk_overlap :] if len(words) > self.chunk_overlap else words
        return [" ".join(overlap)] if overlap else []

    @staticmethod
    def _word_len(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _extract_page_range(text: str) -> tuple[int | None, int | None]:
        matches = re.findall(r"\[Page\s+(\d+)\]", text)
        if not matches:
            return None, None
        pages = [int(m) for m in matches]
        return min(pages), max(pages)
