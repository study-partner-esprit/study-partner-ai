import re

def build_subtopics(sections):
    """
    Build subtopics from full section content with improved processing.
    Features:
    - Meaningful titles from section content
    - Full text summaries (first 2-3 sentences)
    - Enhanced concept extraction
    - Stores full content for tokenization
    """
    subtopics = []
    for idx, sec in enumerate(sections, start=1):
        content_text = " ".join(sec["content"]).strip()
        if not content_text:
            continue  # skip empty sections

        # Generate meaningful title
        if sec["title"] and not sec["title"].startswith('•'):
            title = sec["title"]
        elif content_text:
            # Use first meaningful line as title
            first_line = sec["content"][0] if sec["content"] else f"Subtopic {idx}"
            title = first_line[:100] if len(first_line) > 0 else f"Subtopic {idx}"
        else:
            title = f"Subtopic {idx}"
        
        # Create summary from first 2-3 sentences (not just chars)
        sentences = re.split(r'[.!?]\s+', content_text)
        summary = ". ".join(sentences[:3]).strip()
        if summary and not summary.endswith('.'):
            summary += "."
        
        # Extract key concepts from bullet points and important nouns
        key_concepts = []
        for line in sec["content"]:
            if line.startswith("•") or re.match(r'^\d+\.', line):
                concept = line.lstrip("•").lstrip("0123456789. ").strip()
                if concept and 5 < len(concept) < 60:  # Filter too short/long
                    key_concepts.append(concept)
        
        # Extract capitalized terms (potential concepts)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', content_text)
        for term in capitalized_terms:
            if term not in key_concepts and len(term) > 3:
                key_concepts.append(term)
        
        # Calculate difficulty based on word count and technical terms
        word_count = len(content_text.split())
        technical_indicators = len(re.findall(r'\b(architecture|framework|système|gestion|traitement)\b', content_text.lower()))
        difficulty = min(1.0, (word_count / 200) * 0.5 + (technical_indicators / 5) * 0.5)

        subtopics.append({
            "id": f"sub_{idx}",
            "title": title,
            "summary": summary,
            "full_content": content_text,  # Store full content for tokenization
            "key_concepts": list(set(key_concepts[:10])),  # Deduplicate and limit
            "definitions": [],
            "formulas": [],
            "examples": [],
            "figures": [],
            "tables": [],
            "source_spans": [],
            "difficulty_estimate": round(difficulty, 2),
            "tokenized_chunks": []  # will be filled in tokenization
        })
    
    # Merge similar subtopics with same title pattern
    merged_subtopics = []
    for subtopic in subtopics:
        # Check if we should merge with previous
        if merged_subtopics and should_merge(merged_subtopics[-1], subtopic):
            # Merge content
            merged_subtopics[-1]["full_content"] += " " + subtopic["full_content"]
            merged_subtopics[-1]["summary"] = merged_subtopics[-1]["full_content"][:500] + "..."
            merged_subtopics[-1]["key_concepts"].extend(subtopic["key_concepts"])
            merged_subtopics[-1]["key_concepts"] = list(set(merged_subtopics[-1]["key_concepts"]))[:10]
        else:
            merged_subtopics.append(subtopic)
    
    return merged_subtopics

def should_merge(prev_subtopic, current_subtopic):
    """
    Determine if two subtopics should be merged.
    Merge if:
    - Both have similar titles (e.g., both about "MapReduce")
    - Previous subtopic is very short (<100 words)
    """
    prev_words = len(prev_subtopic["full_content"].split())
    
    # Extract main topic from titles
    prev_main = re.sub(r'^[•\d.]+\s*', '', prev_subtopic["title"]).split(':')[0].strip().lower()
    curr_main = re.sub(r'^[•\d.]+\s*', '', current_subtopic["title"]).split(':')[0].strip().lower()
    
    # Merge if titles are very similar and previous is short
    if prev_words < 100 and prev_main in curr_main or curr_main in prev_main:
        return True
    
    return False
