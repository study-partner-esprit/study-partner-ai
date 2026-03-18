"""
Unit tests for ContentRetriever.add_precomputed_embeddings().

Run with:
    pytest agents/planner/tests/test_retriever.py -v
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so we don't need the real models at test time
# ---------------------------------------------------------------------------


class _FakeVectorStore:
    """Minimal in-memory vector store stub."""

    def __init__(self):
        self.vecs = None
        self.chunks: list[str] = []

    def add(self, embeddings: np.ndarray, chunks: list[str]) -> None:
        self.vecs = embeddings
        self.chunks = chunks

    def search(self, query: np.ndarray, k: int) -> list[str]:
        return self.chunks[:k]


class _FakeEmbedModel:
    """Returns trivial random embeddings."""

    DIM = 8

    def encode(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        vecs = rng.random((len(texts), self.DIM)).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestContentRetrieverPrecomputed:

    def _make_retriever(self):
        from agents.planner.rag.retriever import ContentRetriever

        vs = _FakeVectorStore()
        em = _FakeEmbedModel()
        return ContentRetriever(vs, em), vs

    def test_add_precomputed_stores_chunks(self):
        retriever, vs = self._make_retriever()
        chunks = ["Alpha", "Beta", "Gamma"]
        embeddings = np.random.rand(3, 8).astype("float32")

        n = retriever.add_precomputed_embeddings(chunks, embeddings)

        assert n == 3
        assert retriever.indexed_chunks == chunks
        np.testing.assert_array_equal(vs.vecs, embeddings)

    def test_add_precomputed_list_embeddings(self):
        """Should accept List[List[float]] as well as np.ndarray."""
        retriever, vs = self._make_retriever()
        chunks = ["X", "Y"]
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] * 2

        n = retriever.add_precomputed_embeddings(chunks, embeddings)

        assert n == 2
        assert vs.vecs is not None

    def test_add_precomputed_empty(self):
        retriever, _ = self._make_retriever()
        n = retriever.add_precomputed_embeddings([], [])
        assert n == 0
        assert retriever.indexed_chunks == []

    def test_add_documents_still_works(self):
        """Verify original add_documents path still functions."""
        retriever, vs = self._make_retriever()
        n = retriever.add_documents(["Hello world. " * 20])
        assert n >= 1
        assert len(retriever.indexed_chunks) >= 1

    def test_retrieve_after_precomputed(self):
        """After adding precomputed embeddings, retrieve should return results."""
        retriever, _ = self._make_retriever()
        chunks = ["Machine learning basics", "Neural networks intro", "Python syntax"]
        embeddings = np.random.rand(3, 8).astype("float32")
        retriever.add_precomputed_embeddings(chunks, embeddings)

        results = retriever.retrieve("machine learning", top_k=2)
        # Should return at most 2 non-empty strings
        assert isinstance(results, list)
        assert len(results) <= 2
