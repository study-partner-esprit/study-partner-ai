from pydantic import BaseModel
from typing import List, Optional, Dict


class Subtopic(BaseModel):
    id: str
    title: str
    summary: str
    key_concepts: List[str] = []
    definitions: List[Dict[str, str]] = []
    formulas: List[str] = []
    examples: List[str] = []
    figures: List[str] = []
    tables: List[str] = []
    source_spans: List[Dict[str, str]] = []
    difficulty_estimate: float = 0.0
    tokenized_chunks: List[str] = []


class Topic(BaseModel):
    id: str
    title: str
    summary: str
    subtopics: List[Subtopic]
    prerequisites: List[str] = []
    estimated_learning_time: Optional[int] = None


class CourseKnowledgeJSON(BaseModel):
    course_title: str
    source_files: List[str]
    topics: List[Topic]
    global_prerequisites: List[str] = []
    glossary: List[Dict[str, str]] = []
