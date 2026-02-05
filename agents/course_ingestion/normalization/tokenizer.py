from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of approximately chunk_size words
    with optional overlap between chunks.

    Returns a list of strings.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap  # move start forward
    return chunks


def tokenize_course(
    course_json: Dict, chunk_size: int = 200, overlap: int = 50
) -> Dict:
    """
    Adds a 'tokenized_chunks' field to each subtopic in CourseKnowledgeJSON.
    Each subtopic will contain the chunks of text ready for RAG.
    """
    for topic in course_json.get("topics", []):
        for sub in topic.get("subtopics", []):
            text = sub.get("summary", "")
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            sub["tokenized_chunks"] = chunks
    return course_json


def tokenize_subtopics(
    subtopics: List[Dict], chunk_size: int = 200, overlap: int = 50
) -> List[Dict]:
    """
    Tokenize the full content of each subtopic for RAG.
    Uses the full_content field for complete context, not just summary.
    """
    for sub in subtopics:
        # Use full_content if available, otherwise fall back to summary
        full_text = sub.get("full_content", sub.get("summary", ""))
        if full_text:
            sub["tokenized_chunks"] = chunk_text(
                full_text, chunk_size=chunk_size, overlap=overlap
            )
        else:
            sub["tokenized_chunks"] = []

        # Remove full_content from final output to save space (it's in chunks now)
        if "full_content" in sub:
            del sub["full_content"]

    return subtopics
