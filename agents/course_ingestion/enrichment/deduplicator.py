"""
Semantic deduplication of tokenized chunks.

Removes near-duplicate chunks (cosine-similarity ≥ threshold) from a list
of (chunk, embedding) pairs.  Operates on L2-normalised vectors so
dot-product == cosine similarity.

Usage:
    from agents.course_ingestion.enrichment.deduplicator import deduplicate_chunks
    chunks, embeddings = deduplicate_chunks(chunks, embeddings, threshold=0.95)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_THRESHOLD = 0.95


def deduplicate_chunks(
    chunks: List[str],
    embeddings: List[List[float]] | np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[List[str], List[List[float]]]:
    """
    Remove near-duplicate chunks from *chunks* using cosine similarity.

    Algorithm (greedy, O(N²) worst case, fine for ≤10k chunks):
        1. Keep the first chunk unconditionally.
        2. For each subsequent chunk, compare against all already-kept
           embeddings.  If max similarity ≥ threshold, skip it.

    Args:
        chunks:     Text chunks to deduplicate.
        embeddings: Parallel embeddings (either List[List[float]] or np.ndarray
                    of shape (N, 384)).  Must be L2-normalised.
        threshold:  Cosine-similarity threshold above which a chunk is
                    considered a duplicate (default 0.95).

    Returns:
        (unique_chunks, unique_embeddings) — both as lists for JSON-serialisability.
    """
    if not chunks:
        return [], []

    mat = np.array(embeddings, dtype="float32")  # (N, dim)
    n = len(chunks)

    kept_indices: List[int] = [0]
    kept_mat = mat[0:1]  # shape (1, dim)

    for i in range(1, n):
        vec = mat[i : i + 1]  # (1, dim)
        sims = (kept_mat @ vec.T).flatten()  # dot-product == cosine for normed vecs
        if sims.max() >= threshold:
            continue  # duplicate — skip
        kept_indices.append(i)
        kept_mat = np.vstack([kept_mat, vec])

    original = n
    after = len(kept_indices)
    removed = original - after

    if removed:
        logger.info(
            "deduplication_complete",
            extra={"original": original, "after": after, "removed": removed},
        )

    unique_chunks = [chunks[i] for i in kept_indices]
    unique_embeddings = mat[kept_indices].tolist()
    return unique_chunks, unique_embeddings
