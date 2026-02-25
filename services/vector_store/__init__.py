"""
Vector-store service package.

Public API
----------
    from services.vector_store import get_vector_store, get_embedder
    from services.vector_store import VectorStoreAdapter
"""

from services.vector_store.embedder import get_embedder, EmbeddingService
from services.vector_store.adapter import get_vector_store, VectorStoreAdapter

__all__ = [
    "get_embedder",
    "get_vector_store",
    "EmbeddingService",
    "VectorStoreAdapter",
]
