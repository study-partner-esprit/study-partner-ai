"""
Chunk embedder — embed course subtopic chunks at ingest time.

Called once per course during the ingestion pipeline,
right after tokenize_subtopics() returns.

Usage:
    from agents.course_ingestion.enrichment.chunk_embedder import embed_all_subtopics
    subtopics = embed_all_subtopics(subtopics)
"""

from __future__ import annotations

from typing import List
from services.vector_store.embedder import get_embedder
from utils.logger import get_logger

logger = get_logger(__name__)


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Encode a flat list of text chunks.

    Returns:
        List of float lists (serialisable for MongoDB).
    """
    if not chunks:  # safe: chunks is a plain Python list
        return []
    vecs = get_embedder().encode(chunks)  # returns np.ndarray shape (N, 384)
    if vecs.size == 0:  # safe numpy check — never uses bool(ndarray)
        return []
    return vecs.tolist()


def embed_all_subtopics(subtopics: List[dict]) -> List[dict]:
    """
    Iterate over every subtopic, encode its tokenized_chunks in bulk,
    and attach the embeddings as chunk_embeddings.

    Args:
        subtopics: List of subtopic dicts produced by tokenize_subtopics().
                   Each dict must have a "tokenized_chunks" key.

    Returns:
        The same list with chunk_embeddings populated on every subtopic.
    """
    all_chunks: List[str] = []
    offsets: List[int] = []

    # Collect all chunks from all subtopics in one pass (efficient batching)
    for st in subtopics:
        chunks = st.get("tokenized_chunks", [])
        offsets.append(len(all_chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("embed_all_subtopics_no_chunks")
        return subtopics

    logger.info(
        "embedding_subtopic_chunks",
        extra={"total_chunks": len(all_chunks), "subtopics": len(subtopics)},
    )

    all_vecs = get_embedder().encode(all_chunks).tolist()

    # Distribute embeddings back to each subtopic
    offsets.append(len(all_chunks))  # sentinel
    for i, st in enumerate(subtopics):
        start = offsets[i]
        end = offsets[i + 1]
        st["chunk_embeddings"] = all_vecs[start:end]

    logger.info(
        "embedding_subtopic_chunks_done", extra={"total_chunks": len(all_chunks)}
    )
    return subtopics
