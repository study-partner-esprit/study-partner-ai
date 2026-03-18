from agents.course_ingestion.extraction.pdf_loader import extract_text_from_pdf
from agents.course_ingestion.extraction.ocr import ocr_pdf
from agents.course_ingestion.parsing.layout_parser import detect_sections
from agents.course_ingestion.parsing.section_builder import build_subtopics
from agents.course_ingestion.normalization.normalizer import normalize_course
from agents.course_ingestion.services.database_service import DatabaseService
from agents.course_ingestion.normalization.tokenizer import tokenize_subtopics
from agents.course_ingestion.enrichment.llm_enricher import (
    enrich_subtopic_with_llm,
    generate_subtopic_title,
)
from agents.course_ingestion.enrichment.chunk_embedder import embed_all_subtopics
from agents.course_ingestion.enrichment.deduplicator import deduplicate_chunks
from utils.logger import get_logger

logger = get_logger(__name__)


def ingest_course(course_title: str, pdf_files: list):
    logger.info(
        "ingest_course_start",
        extra={"course_title": course_title, "num_files": len(pdf_files)},
    )
    all_sections = []

    for pdf_path in pdf_files:
        if pdf_path.lower().endswith(".txt"):
            # Read text file directly
            with open(pdf_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            # Step 1: extract text
            text = extract_text_from_pdf(pdf_path)
            # fallback to OCR if needed (text too small)
            if len(text.strip()) < 50:
                logger.info("ingest_ocr_fallback", extra={"file": pdf_path})
                text = ocr_pdf(pdf_path)

        # Step 2: detect sections
        sections = detect_sections(text)
        all_sections.extend(sections)

    # Step 3: build subtopics from sections
    subtopics = build_subtopics(all_sections)

    # Step 4: enrich subtopics with LLM (clean metadata, extract concepts)
    enriched_subtopics = []
    for subtopic in subtopics:
        enriched_data = enrich_subtopic_with_llm(
            subtopic["title"], subtopic["full_content"]
        )
        cleaned_content = enriched_data.get("cleaned_text") or subtopic["full_content"]
        # Update subtopic with enriched content
        subtopic["full_content"] = cleaned_content
        subtopic["key_concepts"] = enriched_data.get(
            "key_concepts", subtopic.get("key_concepts", [])
        )
        subtopic["definitions"] = enriched_data.get("definitions", [])
        subtopic["formulas"] = enriched_data.get("formulas", [])
        subtopic["examples"] = enriched_data.get("examples", [])
        # Re-generate the title from the now-clean content so metadata is gone
        try:
            refined_title = generate_subtopic_title(cleaned_content)
            if refined_title:
                subtopic["title"] = refined_title
        except Exception as title_err:
            logger.warning("title_refinement_failed", extra={"error": str(title_err)})
        enriched_subtopics.append(subtopic)

    # Step 5: tokenize subtopics content
    subtopics = tokenize_subtopics(enriched_subtopics, chunk_size=200, overlap=50)

    # Step 5b: embed all chunks at ingest time and deduplicate per subtopic
    subtopics = embed_all_subtopics(subtopics)
    for st in subtopics:
        chunks = st.get("tokenized_chunks", [])
        embeddings = st.get("chunk_embeddings", [])
        if chunks and embeddings:
            unique_chunks, unique_embeddings = deduplicate_chunks(chunks, embeddings)
            st["tokenized_chunks"] = unique_chunks
            st["chunk_embeddings"] = unique_embeddings

    # Step 6: normalize JSON
    course_json = normalize_course(course_title, subtopics, pdf_files)

    # Step 7: save to MongoDB
    db = DatabaseService()
    course_id = db.save_course(course_json.dict())

    logger.info("ingest_course_done", extra={"course_id": str(course_id)})
    return course_id
