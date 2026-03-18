"""
Embedding generation service.

Central singleton that wraps sentence-transformers all-MiniLM-L6-v2.
Shared across every agent to avoid loading the model multiple times.

Usage:
    from services.vector_store.embedder import get_embedder
    vecs = get_embedder().encode(["hello world"])
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Thin wrapper around sentence-transformers.
    Produces L2-normalised 384-dim float32 vectors so that
    dot-product == cosine similarity in FAISS IndexFlatIP.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIM = 384

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("embedding_model_loading", extra={"model": self.MODEL_NAME})
        self._model = SentenceTransformer(self.MODEL_NAME)
        logger.info("embedding_model_ready", extra={"model": self.MODEL_NAME})

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of strings into float32 numpy vectors.

        Args:
            texts: List of strings to encode.
            batch_size: Inference batch size (default 64).
            normalize: L2-normalise vectors (default True, required for cosine sim).

        Returns:
            np.ndarray of shape (len(texts), 384), dtype float32.
        """
        if not texts:
            return np.empty((0, self.DIM), dtype="float32")

        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return vecs.astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string.  Returns shape (384,)."""
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingService:
    """
    Process-level singleton.  Thread-safe after first call.
    """
    return EmbeddingService()
