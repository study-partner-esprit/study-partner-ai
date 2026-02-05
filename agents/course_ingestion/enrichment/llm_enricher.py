import json
import os
import re
from typing import Dict
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

# Load .env from project root
# enrichment -> course_ingestion -> agents -> study-partner-ai
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        f"GEMINI_API_KEY not found in environment. Checked .env at: {env_path}"
    )

genai.configure(api_key=api_key)


MODEL_NAME = "gemini-2.5-flash"


SYSTEM_PROMPT = """
You are an educational content analyzer specializing in cleaning and structuring learning materials.

Your task is to:
1. CLEAN the content by completely removing ALL non-educational metadata
2. EXTRACT key educational components
3. RETURN properly structured JSON

CRITICAL - You MUST remove ALL of the following:
❌ Email addresses (e.g., professor@university.edu, contact@domain.com)
❌ University/institution names (e.g., "MIT", "Stanford University", "École Polytechnique")
❌ Course codes (e.g., CS101, MATH-202)
❌ Dates and semesters (e.g., "Fall 2025", "January 15", "2024-2025")
❌ Teacher/professor names (e.g., "Prof. Smith", "Dr. Johnson")
❌ Page numbers and headers/footers
❌ Administrative information (office hours, room numbers, contact details)
❌ Copyright notices and legal text
❌ "Chapter X", "Section Y" labels
❌ URLs and website addresses
❌ Document metadata (author, creation date, version)

KEEP ONLY pure educational content:
✓ Core concepts and explanations
✓ Definitions of technical terms
✓ Mathematical formulas and equations
✓ Practical examples demonstrating concepts
✓ Theoretical explanations

Extract into these categories:
- key_concepts: Main ideas and important topics (3-10 items)
- definitions: Technical terms with their precise meanings
- formulas: Mathematical equations, algorithms, or formal notation
- examples: Concrete illustrations of concepts

Return ONLY valid JSON with this exact schema:

{
  "cleaned_text": "Pure educational content with all metadata removed",
  "key_concepts": ["concept1", "concept2"],
  "definitions": [
    {
      "term": "technical term",
      "definition": "clear explanation"
    }
  ],
  "formulas": ["formula or equation"],
  "examples": ["practical example"]
}
"""


def clean_metadata(text: str) -> str:
    """
    Post-process text to remove any remaining metadata that LLM might have missed.
    """
    if not text:
        return text

    # Remove email addresses (multiple patterns for safety)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", text)

    # Remove common academic titles followed by names (extended patterns)
    text = re.sub(
        r"\b(Prof\.|Professor|Dr\.|Docteur|M\.|Mme|Madame|Monsieur|Mr\.|Mrs\.|Ms\.)\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove standalone capitalized names that look like person names (First Last pattern)
    text = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[a-z0-9._%+-]+@", "", text)

    # Remove course codes (e.g., CS101, MATH-202, INF_203)
    text = re.sub(r"\b[A-Z]{2,4}[-_]?\d{2,4}\b", "", text)

    # Remove dates in various formats
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", text)
    text = re.sub(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Janvier|Février|Mars|Avril|Mai|Juin|Juillet|Août|Septembre|Octobre|Novembre|Décembre)\s+\d{1,2},?\s+\d{4}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove page numbers patterns
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bp\.\s*\d+\b", "", text, flags=re.IGNORECASE)

    # Remove common institutional keywords
    text = re.sub(
        r"\b(University|Université|Institute|Institut|School|École|Ecole|College|Collège|ESPRIT|Polytechnique)\s+(of|de|d\')?[A-Z]?[a-z]*(\s+[A-Z][a-z]+)*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove URLs
    text = re.sub(r"https?://[^\s]+", "", text)
    text = re.sub(r"www\.[^\s]+", "", text)

    # Remove chapter/section labels
    text = re.sub(
        r"\b(Chapter|Chapitre|Section|Part|Partie)\s+\d+[:\.]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove "Plan module" and similar course structure words
    text = re.sub(
        r"\b(Plan\s+(module|cours|de\s+cours)|Module\s+plan|Course\s+outline)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Clean up multiple spaces, bullet points, and newlines
    text = re.sub(r"[•·●○■□▪▫]", "", text)  # Remove bullet points
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def enrich_subtopic_with_llm(title: str, text: str) -> Dict:
    """
    Enrich one subtopic using Gemini.
    Pre-cleans input and post-cleans output to ensure metadata removal.
    """

    # Pre-clean the input text before sending to LLM
    cleaned_input = clean_metadata(text)
    cleaned_title = clean_metadata(title)

    model = genai.GenerativeModel(
        model_name=MODEL_NAME, system_instruction=SYSTEM_PROMPT
    )

    prompt = f"""
Analyze and clean the following educational content.

SUBTOPIC TITLE:
{cleaned_title}

RAW CONTENT:
{cleaned_input}

INSTRUCTIONS:
1. Remove ALL emails, university names, dates, professor names, and administrative details
2. Keep ONLY educational content (concepts, explanations, examples)
3. Extract key concepts, definitions, formulas, and examples
4. Return clean, well-structured JSON

Return JSON now:
"""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Gemini sometimes wraps JSON in ```json
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    try:
        data = json.loads(raw)

        # Post-process all text fields to remove any remaining metadata
        if "cleaned_text" in data:
            data["cleaned_text"] = clean_metadata(data["cleaned_text"])

        if "key_concepts" in data and isinstance(data["key_concepts"], list):
            data["key_concepts"] = [
                clean_metadata(c) for c in data["key_concepts"] if c
            ]

        if "definitions" in data and isinstance(data["definitions"], list):
            for defn in data["definitions"]:
                if isinstance(defn, dict):
                    if "term" in defn:
                        defn["term"] = clean_metadata(defn["term"])
                    if "definition" in defn:
                        defn["definition"] = clean_metadata(defn["definition"])

        if "formulas" in data and isinstance(data["formulas"], list):
            data["formulas"] = [clean_metadata(f) for f in data["formulas"] if f]

        if "examples" in data and isinstance(data["examples"], list):
            data["examples"] = [clean_metadata(e) for e in data["examples"] if e]

    except Exception:
        print("LLM enrichment failed, fallback used.")
        print(raw)

        return {
            "cleaned_text": clean_metadata(text),
            "key_concepts": [],
            "definitions": [],
            "formulas": [],
            "examples": [],
        }

    return data
