"""IngestionWorker — consumes `study.ingest.course` jobs (INGEST-06).

Implements the background pipeline that INGEST-05 triggers asynchronously:

    parse → normalize → enrich (LLM) → objectives (BLOOM-04/05)
         → chunk → embed → deduplicate → vector store

Design decisions:

- payload is validated into the strict `IngestionRequest` (INGEST-05); malformed
  input is TERMINAL — retrying cannot fix a bad envelope
- files are read from the shared uploads volume (``INGEST_UPLOADS_DIR``) using
  the storage-relative paths from the envelope; the payload never inlines raw
  content (sandboxed PDF parse is reused from INGEST-04)
- each blocking stage (parsing, enrichment, embedding) runs off the event loop
  via ``asyncio.to_thread`` so the broker loop never blocks (INGEST-06 AC)
- per-stage progress events (parsing/enriching/embedding/indexing) are published
  on ``ai.results`` routing key ``progress`` for INGEST-07
- deduplicated chunks + embeddings land in the FAISS + Mongo vector store via
  ``VectorStoreAdapter.add_course``; replays rebuild the index idempotently,
  so retries (AI-COM-06) never duplicate vectors
- the completed result carries the course JSON (sans embedding vectors — those
  live in the vector store) + pipeline stats; the Node backend persists the
  course document from that payload (INGEST-07)
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from messaging.failures import TerminalError
from workers.base import BaseAIWorker
from workers.schemas import IngestionRequest

logger = logging.getLogger(__name__)

# Default location of the shared uploads volume inside the ai-service container.
UPLOADS_ROOT_DEFAULT = "/app/uploads"


class IngestionWorker(BaseAIWorker):
    job_type = "study.ingest.course"

    def __init__(
        self,
        rabbitmq_url: Optional[str] = None,
        idempotency_store=None,
        prefetch: int = 1,
        uploads_root: Optional[str] = None,
        vector_store=None,
    ) -> None:
        super().__init__(
            rabbitmq_url=rabbitmq_url,
            idempotency_store=idempotency_store,
            prefetch=prefetch,
        )
        # Uploads root is configurable; injected in tests (tmp dirs).
        self._uploads_root = Path(
            uploads_root or os.getenv("INGEST_UPLOADS_DIR", UPLOADS_ROOT_DEFAULT)
        )
        # Vector store is heavy (FAISS + Mongo) → built lazily on first job, or
        # injected in tests.
        self._vector_store = vector_store

    @property
    def vector_store(self):
        if self._vector_store is None:
            from services.vector_store.adapter import get_vector_store

            self._vector_store = get_vector_store()
        return self._vector_store

    # ------------------------------------------------------------------ API

    async def handle(self, payload: Dict[str, Any], envelope) -> Dict[str, Any]:
        request = self._build_request(payload)

        await self._publish_progress(
            envelope,
            stage="parsing",
            progress=0.1,
            detail=f"parsing {len(request.files)} file(s)",
        )

        subtopics, section_count, source_files = await asyncio.to_thread(
            self._parse_stage, request
        )
        if not subtopics:
            raise TerminalError("no subtopics could be derived from the uploaded files")

        await self._publish_progress(
            envelope,
            stage="enriching",
            progress=0.35,
            detail=f"enriching {len(subtopics)} subtopic(s)",
        )
        enriched, objective_stats, classification_stats = await asyncio.to_thread(
            self._enrich_stage, subtopics
        )

        await self._publish_progress(
            envelope, stage="embedding", progress=0.6, detail="chunking and embedding"
        )
        embedded, dedup_removed = await asyncio.to_thread(self._embed_stage, enriched)

        await self._publish_progress(
            envelope,
            stage="indexing",
            progress=0.8,
            detail="upserting chunks into the vector store",
        )
        result = await asyncio.to_thread(
            self._index_stage,
            request.courseId,
            embedded,
            source_files,
            objective_stats,
            classification_stats,
            dedup_removed,
        )

        await self._publish_progress(
            envelope, stage="indexing", progress=1.0, detail="ingestion complete"
        )
        return result

    # ------------------------------------------------------------- stages

    def _parse_stage(self, request: IngestionRequest) -> Tuple[List[dict], int, List[str]]:
        """Extract text from every file, split into sections, build subtopics.

        Returns (subtopics, section_count, source_files) — pure CPU/IO, runs off
        the event loop.
        """
        from agents.course_ingestion.parsing.layout_parser import detect_sections
        from agents.course_ingestion.parsing.section_builder import build_subtopics

        source_files: List[str] = []
        all_sections = []
        for meta in request.files:
            path = self._resolve_path(meta, request.fileRef)
            text = self._extract_text(meta, path)
            source_files.append(str(path))
            all_sections.extend(detect_sections(text))

        if not all_sections:
            logger.warning("ingest_no_sections", extra={"courseId": request.courseId})
        return build_subtopics(all_sections), len(all_sections), source_files

    def _extract_text(self, meta, path: Path) -> str:
        """Read a staged file. PDFs go through the INGEST-04 sandboxed parser
        with OCR fallback; text files are read as UTF-8 directly."""
        from agents.course_ingestion.extraction.pdf_loader import (
            EncryptedPdfError,
            PdfParseLimitError,
            PdfParseTimeoutError,
            PdfValidationError,
            extract_text_from_pdf_sandboxed,
        )

        is_pdf = (meta.mimetype or "").lower() == "application/pdf" or meta.filename.lower().endswith(
            ".pdf"
        )
        if not is_pdf:
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                # Data problem: retrying cannot fix an undecodable file.
                raise TerminalError(f"text file is not valid UTF-8: {meta.filename}") from exc

        try:
            text = extract_text_from_pdf_sandboxed(str(path))
        except (PdfParseTimeoutError, PdfParseLimitError):
            # Named first: they subclass PdfValidationError but are transient
            # resource pressure → retryable, never terminal.
            raise
        except (EncryptedPdfError, PdfValidationError) as exc:
            # The upload flunks what the sandbox can parse — data, not infra.
            raise TerminalError(f"PDF rejected by the sandbox parser: {exc}") from exc

        # Scanned/imaged PDFs produce little or no text → OCR fallback. Imported
        # lazily: pytesseract/poppler are heavy and only needed on this path.
        if len(text.strip()) < 50:
            from agents.course_ingestion.extraction.ocr import ocr_pdf

            logger.info("ingest_ocr_fallback", extra={"file": path.name})
            text = ocr_pdf(str(path))
        return text

    def _resolve_path(self, meta, fileRef: str) -> Path:
        """Map an envelope file entry onto the uploads volume with containment.

        ``files[].path`` is relative to the uploads root (e.g.
        ``courses/<courseId>/<filename>``); ``fileRef`` is ``uploads/courses/
        <courseId>``. A missing volume/file is terminal (data, not transient).
        """
        root = self._uploads_root.resolve()
        if not root.is_dir():
            raise TerminalError(f"uploads root not mounted: {root}")

        file_ref = fileRef.lstrip("/")
        if file_ref.startswith("uploads/"):
            file_ref = file_ref[len("uploads/"):]
        rel = (meta.path or os.path.join(file_ref, meta.filename)).replace("\\", "/")
        path = (root / rel).resolve()

        if path != root and root not in path.parents:
            raise TerminalError(f"file path escapes the uploads root: {meta.filename}")
        if not path.is_file():
            raise TerminalError(f"ingestion source file not found: {meta.filename}")
        return path

    def _enrich_stage(
        self, subtopics: List[dict]
    ) -> Tuple[List[dict], Dict[str, Any], Dict[str, Any]]:
        """LLM enrichment + BLOOM-04 objectives + BLOOM-05 classification."""
        from agents.course_ingestion.enrichment.llm_enricher import (
            enrich_subtopic_with_llm,
            generate_subtopic_title,
        )
        from agents.course_ingestion.enrichment.objective_extractor import (
            extract_objectives_for_document,
        )
        from bloom.classifier import classify_objectives_for_document

        enriched = []
        for st in subtopics:
            enriched_data = enrich_subtopic_with_llm(st["title"], st["full_content"])
            cleaned = enriched_data.get("cleaned_text") or st["full_content"]
            st["full_content"] = cleaned
            st["key_concepts"] = enriched_data.get("key_concepts", st.get("key_concepts", []))
            st["definitions"] = enriched_data.get("definitions", [])
            st["formulas"] = enriched_data.get("formulas", [])
            st["examples"] = enriched_data.get("examples", [])
            try:
                refined_title = generate_subtopic_title(cleaned)
                if refined_title:
                    st["title"] = refined_title
            except Exception as title_err:  # title refinement is best-effort
                logger.warning("ingest_title_refinement_failed", extra={"error": str(title_err)})
            enriched.append(st)

        objective_stats = extract_objectives_for_document(enriched)
        classification_stats = classify_objectives_for_document(enriched)
        return enriched, objective_stats, classification_stats

    def _embed_stage(self, subtopics: List[dict]) -> Tuple[List[dict], int]:
        """Chunk (tokenize), embed, then deduplicate near-identical chunks."""
        from agents.course_ingestion.enrichment.chunk_embedder import embed_all_subtopics
        from agents.course_ingestion.enrichment.deduplicator import deduplicate_chunks
        from agents.course_ingestion.normalization.tokenizer import tokenize_subtopics

        tokenized = tokenize_subtopics(subtopics, chunk_size=200, overlap=50)
        embedded = embed_all_subtopics(tokenized)

        removed = 0
        for st in embedded:
            chunks = st.get("tokenized_chunks", [])
            embeddings = st.get("chunk_embeddings", [])
            if not chunks or not embeddings:
                continue
            unique_chunks, unique_embeddings = deduplicate_chunks(chunks, embeddings)
            removed += len(chunks) - len(unique_chunks)
            st["tokenized_chunks"] = unique_chunks
            st["chunk_embeddings"] = unique_embeddings
        logger.info(
            "ingest_dedup_done",
            extra={"removedChunks": removed, "remainingChunks": sum(
                len(st.get("tokenized_chunks", [])) for st in embedded
            )},
        )
        return embedded, removed

    def _index_stage(
        self,
        course_id: str,
        subtopics: List[dict],
        source_files: List[str],
        objective_stats: Dict[str, Any],
        classification_stats: Dict[str, Any],
        dedup_removed: int,
    ) -> Dict[str, Any]:
        """Normalize into course JSON, flatten chunk records, index the vector
        store. Returns the completed-result payload for the Node backend."""
        from agents.course_ingestion.normalization.normalizer import normalize_course

        course_json = normalize_course(course_id, subtopics, source_files)
        chunks, embeddings, metadatas = _flatten_chunk_records(course_id, course_json)

        indexed = False
        if chunks and embeddings.size:
            self.vector_store.add_course(course_id, chunks, embeddings, metadatas)
            indexed = True

        stats = {
            "files": len(source_files),
            "subtopics": len(subtopics),
            "chunks": len(chunks),
            "indexedChunks": len(chunks) if indexed else 0,
            "deduplicatedChunks": dedup_removed,
            "objectivesExtracted": int(objective_stats.get("extracted", 0)),
            "objectivesClassified": int(classification_stats.get("classified", 0)),
            "objectivesNeedingReview": int(classification_stats.get("needs_review", 0)),
            "vectorStoreIndexed": indexed,
        }
        logger.info(
            "ingest_completed",
            extra={"courseId": course_id, "stats": stats},
        )
        return {
            "courseId": course_id,
            "status": "completed",
            "stats": stats,
            "course": _strip_vectors(course_json.model_dump(mode="json")),
        }

    # ------------------------------------------------------------ internals

    def _build_request(self, payload: Any) -> IngestionRequest:
        if not isinstance(payload, dict):
            raise TerminalError("invalid ingest payload: must be an object")
        try:
            return IngestionRequest.model_validate(payload)
        except ValidationError as exc:
            raise TerminalError(f"invalid ingest payload: {exc}") from exc


def _flatten_chunk_records(
    course_id: str, course_json
) -> Tuple[List[str], Any, List[Dict[str, Any]]]:
    """Flatten per-subtopic tokenized chunks + embeddings into index-ready arrays.

    Returns (chunks, np.ndarray float32 (N, 384) embeddings, metadatas). Each
    metadata record carries the topic/subtopic it came from so the vector store
    can scope search results back to course structure.
    """
    import numpy as np

    chunks: List[str] = []
    vectors: List[Any] = []
    metadatas: List[Dict[str, Any]] = []
    for topic in course_json.topics:
        for sub in topic.subtopics:
            sub_chunks = sub.tokenized_chunks or []
            sub_vecs = sub.chunk_embeddings or []
            for idx, chunk in enumerate(sub_chunks):
                if idx >= len(sub_vecs):
                    continue
                chunks.append(chunk)
                vectors.append(sub_vecs[idx])
                metadatas.append(
                    {
                        "course_id": course_id,
                        "topic_id": topic.id,
                        "topic_title": topic.title,
                        "subtopic_id": sub.id,
                        "subtopic_title": sub.title,
                    }
                )
    embeddings = np.asarray(vectors, dtype="float32") if vectors else np.zeros((0, 0), dtype="float32")
    return chunks, embeddings, metadatas


def _strip_vectors(course: dict) -> dict:
    """Remove heavy vector data from the result payload — chunks/embeddings live
    in the vector store, not in ai.results events."""
    for topic in course.get("topics", []):
        for sub in topic.get("subtopics", []):
            sub.pop("tokenized_chunks", None)
            sub.pop("chunk_embeddings", None)
    return course


if __name__ == "__main__":  # pragma: no cover - manual run entrypoint
    logging.basicConfig(level=logging.INFO)
    IngestionWorker().run()