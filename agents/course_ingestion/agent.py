from agents.course_ingestion.extraction.pdf_loader import extract_text_from_pdf
from agents.course_ingestion.extraction.ocr import ocr_pdf
from agents.course_ingestion.parsing.layout_parser import detect_sections
from agents.course_ingestion.parsing.section_builder import build_subtopics
from agents.course_ingestion.normalization.normalizer import normalize_course
from agents.course_ingestion.services.database_service import DatabaseService
from agents.course_ingestion.normalization.tokenizer import tokenize_subtopics
from agents.course_ingestion.enrichment.llm_enricher import enrich_subtopic_with_llm


def ingest_course(course_title: str, pdf_files: list):
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
                text = ocr_pdf(pdf_path)

        # Step 2: detect sections
        sections = detect_sections(text)
        all_sections.extend(sections)

    # Step 3: build subtopics from sections
    subtopics = build_subtopics(all_sections)

    # Step 4: enrich subtopics with LLM (clean metadata, extract concepts)
    enriched_subtopics = []
    for subtopic in subtopics:
        enriched_data = enrich_subtopic_with_llm(subtopic['title'], subtopic['full_content'])
        # Update subtopic with enriched content
        subtopic['full_content'] = enriched_data.get('cleaned_text', subtopic['full_content'])
        subtopic['key_concepts'] = enriched_data.get('key_concepts', subtopic.get('key_concepts', []))
        subtopic['definitions'] = enriched_data.get('definitions', [])
        subtopic['formulas'] = enriched_data.get('formulas', [])
        subtopic['examples'] = enriched_data.get('examples', [])
        enriched_subtopics.append(subtopic)

    # Step 5: tokenize subtopics content
    subtopics = tokenize_subtopics(enriched_subtopics, chunk_size=200, overlap=50)

    # Step 6: normalize JSON
    course_json = normalize_course(course_title, subtopics, pdf_files)

    # Step 7: save to MongoDB
    db = DatabaseService()
    course_id = db.save_course(course_json.dict())

    return course_id
