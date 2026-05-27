from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from ragstarter.core.config import Settings


class EmbeddingService:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = SentenceTransformer(model_name_or_path=model_name, device=device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


@lru_cache(maxsize=2)
def get_embedding_service(model_name: str, device: str) -> EmbeddingService:
    return EmbeddingService(model_name=model_name, device=device)


def build_embedding_service(settings: Settings) -> EmbeddingService:
    return get_embedding_service(settings.embedding_model, settings.embedding_device)
