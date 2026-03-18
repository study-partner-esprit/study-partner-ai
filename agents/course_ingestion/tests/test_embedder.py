"""
Unit tests for chunk_embedder and deduplicator.

Run with:
    pytest agents/course_ingestion/tests/test_embedder.py -v
"""

import numpy as np
import pytest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normed(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-8)


def _fake_encode(texts):
    """Deterministic fake encoder — returns unit-normalised random vectors."""
    rng = np.random.default_rng(seed=sum(len(t) for t in texts))
    vecs = _normed(rng.random((len(texts), 4)).astype("float32"))
    return vecs


# ---------------------------------------------------------------------------
# Tests: deduplicator
# ---------------------------------------------------------------------------


class TestDeduplicator:
    from agents.course_ingestion.enrichment.deduplicator import deduplicate_chunks

    def test_no_duplicates_unchanged(self):
        from agents.course_ingestion.enrichment.deduplicator import deduplicate_chunks

        chunks = ["Alpha", "Beta", "Gamma"]
        # Orthogonal unit vectors — all pairwise sims are ~0
        vecs = _normed(np.eye(3, 4, dtype="float32"))
        out_chunks, out_vecs = deduplicate_chunks(chunks, vecs, threshold=0.95)
        assert out_chunks == chunks
        assert len(out_vecs) == 3

    def test_exact_duplicate_removed(self):
        from agents.course_ingestion.enrichment.deduplicator import deduplicate_chunks

        chunks = ["A", "B", "A_copy"]
        # First and last are identical vectors
        v = _normed(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype="float32")
        )
        out_chunks, out_vecs = deduplicate_chunks(chunks, v, threshold=0.95)
        assert len(out_chunks) == 2
        assert "A" in out_chunks
        assert "B" in out_chunks

    def test_empty_input(self):
        from agents.course_ingestion.enrichment.deduplicator import deduplicate_chunks

        assert deduplicate_chunks([], []) == ([], [])

    def test_single_element(self):
        from agents.course_ingestion.enrichment.deduplicator import deduplicate_chunks

        v = _normed(np.array([[1, 0, 0, 0]], dtype="float32"))
        out_c, out_v = deduplicate_chunks(["X"], v)
        assert out_c == ["X"]
        assert len(out_v) == 1

    def test_list_input_accepted(self):
        from agents.course_ingestion.enrichment.deduplicator import deduplicate_chunks

        # List[List[float]] format
        chunks = ["A", "B"]
        embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        out_c, out_v = deduplicate_chunks(chunks, embeddings, threshold=0.95)
        assert len(out_c) == 2


# ---------------------------------------------------------------------------
# Tests: chunk_embedder
# ---------------------------------------------------------------------------


class TestChunkEmbedder:

    def test_embed_all_subtopics_attaches_embeddings(self):
        fake_embedder = type(
            "FakeEmbedder",
            (),
            {"encode": lambda self, texts, **kw: _fake_encode(texts)},
        )()

        with patch(
            "agents.course_ingestion.enrichment.chunk_embedder.get_embedder",
            return_value=fake_embedder,
        ):
            from agents.course_ingestion.enrichment.chunk_embedder import (
                embed_all_subtopics,
            )

            subtopics = [
                {"tokenized_chunks": ["chunk A", "chunk B"]},
                {"tokenized_chunks": ["chunk C"]},
            ]
            result = embed_all_subtopics(subtopics)

        assert len(result[0]["chunk_embeddings"]) == 2
        assert len(result[1]["chunk_embeddings"]) == 1
        # Each embedding is a list of floats
        assert isinstance(result[0]["chunk_embeddings"][0], list)

    def test_embed_all_no_chunks_returns_unchanged(self):
        fake_embedder = type(
            "FakeEmbedder",
            (),
            {"encode": lambda self, texts, **kw: _fake_encode(texts)},
        )()

        with patch(
            "agents.course_ingestion.enrichment.chunk_embedder.get_embedder",
            return_value=fake_embedder,
        ):
            from agents.course_ingestion.enrichment.chunk_embedder import (
                embed_all_subtopics,
            )

            subtopics = [{"tokenized_chunks": []}]
            result = embed_all_subtopics(subtopics)
        # Should return unmodified (no embeddings key added)
        assert "chunk_embeddings" not in result[0]
