import re

def detect_sections(text: str):
    """
    Detects headings and groups lines under headings with improved heuristics.
    Major heading patterns:
    - Lines starting with 'Chapitre', 'Section', or numbers followed by dot
    - Lines in ALL CAPS (at least 3 words)
    - Technical terms followed by ':'
    Ignores minor bullet points and dates/page numbers.
    """
    lines = text.split("\n")
    sections = []
    current_section = {"title": "", "content": []}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip page numbers and dates
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', line) or re.match(r'^\d+$', line):
            continue
        
        # Major heading heuristics
        is_major_heading = (
            re.match(r'^(Chapitre|CHAPITRE|Section|SECTION)\s+', line, re.IGNORECASE) or
            re.match(r'^\d+\.\s+[A-Z]', line) or  # Numbered sections like "1. Introduction"
            (line.isupper() and len(line.split()) >= 2) or  # ALL CAPS with at least 2 words
            (re.match(r'^[A-Z][^:]{2,}:$', line) and len(line.split()) <= 5)  # Title case ending with ':'
        )
        
        # Ignore minor bullet points as headings
        is_bullet = line.startswith('•') or re.match(r'^[a-z]', line)
        
        if is_major_heading and not is_bullet:
            if current_section["title"] or current_section["content"]:
                sections.append(current_section)
            current_section = {"title": line, "content": []}
        else:
            current_section["content"].append(line)

    if current_section["title"] or current_section["content"]:
        sections.append(current_section)

    # Merge small sections (<50 words) with previous section
    merged_sections = []
    for section in sections:
        content_text = " ".join(section["content"])
        word_count = len(content_text.split())
        
        if merged_sections and word_count < 50 and section["title"].startswith('•'):
            # Merge with previous section
            merged_sections[-1]["content"].append(section["title"])
            merged_sections[-1]["content"].extend(section["content"])
        else:
            merged_sections.append(section)
    
    return merged_sections
