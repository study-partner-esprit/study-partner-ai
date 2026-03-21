"""
FAISS index persistence helpers — save / load / delete per-course indices.

Separate from VectorStoreAdapter so the Planner can also manage its own
per-request in-memory index without requiring the full service layer.

Usage:
    from agents.planner.rag.index_store import save_index, load_index
    save_index(vector_store.index, vector_store.chunks, course_id)
    index, chunks = load_index(course_id)  # (None, None) if not found
"""

from __future__ import annotations

import json
import os
import faiss
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "/tmp/study_partner_faiss"))


def _ensure_dir() -> None:
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def save_index(
    index: faiss.Index,
    chunks: List[str],
    course_id: str,
    embeddings: Optional[np.ndarray] = None,
) -> None:
    """
    Write *index* and its associated *chunks* to disk.

    Args:
        index:      FAISS index containing the chunk vectors.
        chunks:     Parallel list of text chunks.
        course_id:  Used as the file stem (slug-safe string recommended).
        embeddings: Optional — raw numpy matrix stored as .npy alongside the
                    index so that other agents can load pre-computed vectors.
    """
    _ensure_dir()
    index_path = FAISS_INDEX_DIR / f"{course_id}.faiss"
    meta_path = FAISS_INDEX_DIR / f"{course_id}.meta.json"

    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps({"chunks": chunks}, ensure_ascii=False))

    if embeddings is not None:
        np.save(str(FAISS_INDEX_DIR / f"{course_id}.npy"), embeddings)

    logger.info(
        "index_store_saved",
        extra={"course_id": course_id, "ntotal": index.ntotal},
    )


def load_index(course_id: str) -> Tuple[Optional[faiss.Index], Optional[List[str]]]:
    """
    Load an index from disk.

    Returns:
        (index, chunks) or (None, None) if not found.
    """
    _ensure_dir()
    index_path = FAISS_INDEX_DIR / f"{course_id}.faiss"
    meta_path = FAISS_INDEX_DIR / f"{course_id}.meta.json"

    if not (index_path.exists() and meta_path.exists()):
        return None, None

    try:
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text())
        chunks: List[str] = meta.get("chunks", [])
        if index.ntotal != len(chunks):
            logger.warning(
                "index_store_integrity_mismatch",
                extra={"course_id": course_id, "ntotal": index.ntotal, "chunks": len(chunks)},
            )
            return None, None
        logger.info(
            "index_store_loaded",
            extra={"course_id": course_id, "ntotal": index.ntotal},
        )
        return index, chunks
    except Exception as exc:
        logger.warning(
            "index_store_load_error",
            extra={"course_id": course_id, "error": str(exc)},
        )
        return None, None


def load_embeddings(course_id: str) -> Optional[np.ndarray]:
    """Load the raw .npy embeddings if saved alongside the index."""
    _ensure_dir()
    npy_path = FAISS_INDEX_DIR / f"{course_id}.npy"
    if not npy_path.exists():
        return None
    try:
        return np.load(str(npy_path))
    except Exception as exc:
        logger.warning(
            "index_store_npy_load_error",
            extra={"course_id": course_id, "error": str(exc)},
        )
        return None


def rebuild_index_from_embeddings(
    course_id: str,
    chunks: List[str],
    embeddings: np.ndarray,
) -> Tuple[Optional[faiss.Index], Optional[List[str]]]:
    """Rebuild and persist a FAISS index from raw embeddings for recovery flows."""
    try:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if len(chunks) == 0:
            raise ValueError("cannot rebuild empty index")

        dim = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype("float32"))
        save_index(index=index, chunks=chunks, course_id=course_id, embeddings=embeddings)
        return index, chunks
    except Exception as exc:
        logger.warning(
            "index_store_rebuild_failed",
            extra={"course_id": course_id, "error": str(exc)},
        )
        return None, None


def delete_index(course_id: str) -> None:
    """Remove all index files for *course_id* from disk."""
    _ensure_dir()
    for suffix in (".faiss", ".meta.json", ".npy"):
        p = FAISS_INDEX_DIR / f"{course_id}{suffix}"
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    logger.info("index_store_deleted", extra={"course_id": course_id})
