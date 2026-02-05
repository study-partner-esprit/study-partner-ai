from extraction.pdf_loader import extract_text_from_pdf
from extraction.ocr import ocr_pdf
from parsing.layout_parser import detect_sections
from parsing.section_builder import build_subtopics
from normalization.normalizer import normalize_course
from services.database_service import DatabaseService
from normalization.tokenizer import tokenize_subtopics

def ingest_course(course_title: str, pdf_files: list):
    all_sections = []

    for pdf_path in pdf_files:
        if pdf_path.lower().endswith('.txt'):
            # Read text file directly
            with open(pdf_path, 'r', encoding='utf-8') as f:
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

    # Step 4: tokenize subtopics content
    subtopics = tokenize_subtopics(subtopics, chunk_size=200, overlap=50)

    # Step 5: normalize JSON
    course_json = normalize_course(course_title, subtopics, pdf_files)

    # Step 6: save to MongoDB
    db = DatabaseService()
    course_id = db.save_course(course_json.dict())

    return course_id
