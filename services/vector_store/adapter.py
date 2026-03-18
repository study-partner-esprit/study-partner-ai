"""
Vector store adapter — FAISS IndexFlatIP backed by MongoDB embedding storage.

Responsibilities:
  - Store raw chunk vectors in MongoDB (collection: chunk_embeddings)
  - Build an in-memory FAISS IndexFlatIP per course
  - Persist FAISS indices to disk (FAISS_INDEX_DIR)
  - Search: encode a query → retrieve top-k chunks with metadata

Usage:
    from services.vector_store.adapter import get_vector_store
    vs = get_vector_store()
    vs.add_course(course_id, chunks, embeddings, metadatas)
    results = vs.search(course_id, query_text, k=5)
"""

from __future__ import annotations

import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

from services.vector_store.embedder import get_embedder
from utils.logger import get_logger

logger = get_logger(__name__)

FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "/tmp/study_partner_faiss"))
CHUNK_EMBED_COLLECTION = "chunk_embeddings"


class VectorStoreAdapter:
    """
    Per-course FAISS index with MongoDB persistence for the raw embeddings.

    Index layout:
        _indices[course_id] = {
            "index": faiss.Index,
            "chunks": list[str],
            "metas":  list[dict],
        }
    """

    DIM = 384

    def __init__(self, db) -> None:
        """
        Args:
            db: pymongo Database object (pass None for test / offline mode).
        """
        self._db = db
        self._indices: Dict[str, Dict[str, Any]] = {}
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add_course(
        self,
        course_id: str,
        chunks: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Build (or rebuild) the FAISS index for *course_id* from pre-computed embeddings.

        Side-effects:
          - Persists embeddings to MongoDB (upsert by course_id).
          - Saves the FAISS index to FAISS_INDEX_DIR/<course_id>.faiss + .meta.json.

        Args:
            course_id:  Unique course identifier.
            chunks:     Parallel list of text chunks.
            embeddings: np.ndarray shape (N, 384), float32, L2-normalised.
            metadatas:  Optional per-chunk metadata dicts.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        metadatas = metadatas or [{} for _ in chunks]
        n = len(chunks)

        # Build FAISS index
        index = faiss.IndexFlatIP(self.DIM)
        index.add(embeddings.astype("float32"))

        self._indices[course_id] = {
            "index": index,
            "chunks": chunks,
            "metas": metadatas,
        }

        # Persist to disk
        self._save_index(course_id)

        # Persist raw embeddings to MongoDB
        if self._db is not None:
            self._save_embeddings_to_mongo(course_id, chunks, embeddings, metadatas)

        logger.info(
            "vector_store_course_indexed",
            extra={"course_id": course_id, "num_chunks": n},
        )

    def add_texts(
        self,
        course_id: str,
        chunks: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Encode *chunks* on-the-fly, store in the index and return the embeddings.

        Useful for agents that do not have pre-computed embeddings.
        """
        embedder = get_embedder()
        embeddings = embedder.encode(chunks)
        self.add_course(course_id, chunks, embeddings, metadatas)
        return embeddings

    def search(
        self,
        course_id: str,
        query: str | np.ndarray,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-*k* most similar chunks for a *query*.

        Args:
            course_id: Course to search within.
            query:     Either a plain text string (will be encoded) or a
                       float32 vector of shape (384,).
            k:         Number of results to return.

        Returns:
            List of dicts with keys: chunk, score, meta, rank.
        """
        entry = self._ensure_loaded(course_id)
        if entry is None:
            return []

        index: faiss.Index = entry["index"]
        chunks: List[str] = entry["chunks"]
        metas: List[dict] = entry["metas"]

        if isinstance(query, str):
            q_vec = get_embedder().encode_one(query)
        else:
            q_vec = query.astype("float32")

        q_vec = q_vec.reshape(1, -1)

        k = min(k, index.ntotal)
        if k == 0:
            return []

        scores, idxs = index.search(q_vec, k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0])):
            if idx == -1:
                continue
            results.append(
                {
                    "chunk": chunks[idx],
                    "score": float(score),
                    "meta": metas[idx],
                    "rank": rank,
                }
            )
        return results

    def load_course(self, course_id: str) -> bool:
        """
        Explicitly load a course index from disk / MongoDB.

        Returns True if loaded successfully, False if not found.
        """
        return self._ensure_loaded(course_id) is not None

    def delete_course(self, course_id: str) -> None:
        """Remove the FAISS index from memory and disk."""
        self._indices.pop(course_id, None)
        index_path = FAISS_INDEX_DIR / f"{course_id}.faiss"
        meta_path = FAISS_INDEX_DIR / f"{course_id}.meta.json"
        for p in (index_path, meta_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        logger.info("vector_store_course_deleted", extra={"course_id": course_id})

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self, course_id: str) -> Optional[Dict[str, Any]]:
        if course_id in self._indices:
            return self._indices[course_id]

        # Try disk
        if self._load_index_from_disk(course_id):
            return self._indices[course_id]

        # Try MongoDB
        if self._db is not None and self._load_index_from_mongo(course_id):
            return self._indices[course_id]

        logger.warning("vector_store_course_not_found", extra={"course_id": course_id})
        return None

    # -- Disk I/O -------------------------------------------------------- #

    def _save_index(self, course_id: str) -> None:
        index_path = FAISS_INDEX_DIR / f"{course_id}.faiss"
        meta_path = FAISS_INDEX_DIR / f"{course_id}.meta.json"
        entry = self._indices[course_id]
        faiss.write_index(entry["index"], str(index_path))
        meta = {"chunks": entry["chunks"], "metas": entry["metas"]}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False))

    def _load_index_from_disk(self, course_id: str) -> bool:
        index_path = FAISS_INDEX_DIR / f"{course_id}.faiss"
        meta_path = FAISS_INDEX_DIR / f"{course_id}.meta.json"
        if not (index_path.exists() and meta_path.exists()):
            return False
        try:
            index = faiss.read_index(str(index_path))
            meta = json.loads(meta_path.read_text())
            self._indices[course_id] = {
                "index": index,
                "chunks": meta["chunks"],
                "metas": meta["metas"],
            }
            logger.info(
                "vector_store_loaded_from_disk",
                extra={"course_id": course_id, "num_chunks": index.ntotal},
            )
            return True
        except Exception as exc:
            logger.warning(
                "vector_store_disk_load_failed",
                extra={"course_id": course_id, "error": str(exc)},
            )
            return False

    # -- MongoDB I/O ----------------------------------------------------- #

    def _save_embeddings_to_mongo(
        self,
        course_id: str,
        chunks: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
    ) -> None:
        col = self._db[CHUNK_EMBED_COLLECTION]
        doc = {
            "course_id": course_id,
            "chunks": chunks,
            "metas": metadatas,
            "embeddings": embeddings.tolist(),
        }
        col.update_one({"course_id": course_id}, {"$set": doc}, upsert=True)

    def _load_index_from_mongo(self, course_id: str) -> bool:
        try:
            col = self._db[CHUNK_EMBED_COLLECTION]
            doc = col.find_one({"course_id": course_id})
            if doc is None:
                return False
            chunks = doc["chunks"]
            metas = doc["metas"]
            embeddings = np.array(doc["embeddings"], dtype="float32")

            index = faiss.IndexFlatIP(self.DIM)
            index.add(embeddings)
            self._indices[course_id] = {
                "index": index,
                "chunks": chunks,
                "metas": metas,
            }
            # Cache to disk for next restart
            self._save_index(course_id)
            logger.info(
                "vector_store_loaded_from_mongo",
                extra={"course_id": course_id, "num_chunks": len(chunks)},
            )
            return True
        except Exception as exc:
            logger.warning(
                "vector_store_mongo_load_failed",
                extra={"course_id": course_id, "error": str(exc)},
            )
            return False


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreAdapter:
    """
    Process-level singleton.  Imports MongoDB connection lazily to avoid
    circular imports at module load time.
    """
    from services.database import get_db  # noqa: PLC0415

    db = None
    try:
        db = get_db()
    except Exception:
        logger.warning("vector_store_no_db", extra={"reason": "could not connect"})
    return VectorStoreAdapter(db)
