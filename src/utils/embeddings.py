"""Embedding utility functions."""

from __future__ import annotations

from typing import List

from config import settings


class EmbeddingModel:
    """Wrapper around sentence-transformers for local embeddings."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
        return cls._instance

    def _get_model(self):
        """Load the sentence transformer lazily on first use."""

        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                settings.embedding_model,
                device="cpu",
            )
        return self.model

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        return self._get_model().encode(texts, convert_to_tensor=False).tolist()

    def encode_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.encode([text])[0]
