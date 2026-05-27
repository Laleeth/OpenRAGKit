from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from ragstarter.core.config import Settings
from ragstarter.services.vector_store import VectorHit

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    name: str

    def generate(self, question: str, context: str) -> str: ...


@dataclass(slots=True)
class OpenAICompatibleProvider:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 60.0
    name: str = "openai_compatible"

    def generate(self, question: str, context: str) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        system = (
            "You are a careful RAG assistant. Answer only from the provided context. "
            "If the context is insufficient, say what is missing. Use short citations like [1], [2]."
        )
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        "Context:\n"
                        f"{context}\n\n"
                        "Question:\n"
                        f"{question}\n\n"
                        "Answer with clear citations."
                    ),
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"].strip()


@dataclass(slots=True)
class OllamaProvider:
    base_url: str
    model: str
    timeout_s: float = 60.0
    name: str = "ollama"

    def generate(self, question: str, context: str) -> str:
        url = self.base_url.rstrip("/") + "/api/generate"
        prompt = (
            "You are a careful RAG assistant. Answer only from the provided context. "
            "If the context is insufficient, say what is missing. Use short citations like [1], [2].\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        return str(data.get("response", "")).strip()


@dataclass(slots=True)
class ExtractiveProvider:
    name: str = "extractive"

    def generate(self, question: str, context: str) -> str:
        # context format: "[1] ...\ntext"
        chunks = self._parse_context(context)
        if not chunks:
            return "I could not find any relevant context."

        scored: list[tuple[float, int, str]] = []
        q_tokens = set(self._tokenize(question))
        for idx, (label, text) in enumerate(chunks):
            score = self._score(text, q_tokens)
            scored.append((score, idx, f"[{label}] {self._best_excerpt(text, q_tokens)}"))
        scored.sort(key=lambda x: (-x[0], x[1]))
        best = [s for s in scored[:3] if s[0] > 0]
        if not best:
            best = scored[:2]

        answer = "Here is the most relevant evidence I found:\n"
        for score, _, snippet in best:
            answer += f"- {snippet}\n"
        answer += "\nThis answer is extracted from the indexed sources. For a richer generated response, set an OpenAI-compatible or Ollama provider."
        return answer.strip()

    def _parse_context(self, context: str) -> list[tuple[str, str]]:
        chunks: list[tuple[str, str]] = []
        parts = re.split(r"\n(?=\[\d+\]\s)", context.strip())
        for part in parts:
            m = re.match(r"^\[(\d+)\]\s*(.*)$", part.strip(), re.S)
            if m:
                chunks.append((m.group(1), m.group(2).strip()))
        return chunks

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _score(self, text: str, q_tokens: set[str]) -> float:
        tokens = self._tokenize(text)
        if not tokens or not q_tokens:
            return 0.0
        overlap = len(set(tokens) & q_tokens)
        return overlap / math.sqrt(len(set(tokens)) + 1)

    def _best_excerpt(self, text: str, q_tokens: set[str], max_words: int = 40) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        best = sentences[0] if sentences else text[:300]
        best_score = -1.0
        for sent in sentences:
            tokens = set(self._tokenize(sent))
            score = len(tokens & q_tokens)
            if score > best_score:
                best_score = score
                best = sent
        words = best.split()
        if len(words) > max_words:
            best = " ".join(words[:max_words]) + "..."
        return best


def build_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower().strip()
    if provider == "openai_compatible" and settings.llm_api_key:
        return OpenAICompatibleProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
        )
    if provider == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
        )
    return ExtractiveProvider()
