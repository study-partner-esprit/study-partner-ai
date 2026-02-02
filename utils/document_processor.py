"""Document processor for PDF text extraction and LLM-based JSON normalization."""
import json
import re
from typing import Dict, Any, List
import PyPDF2
import io
import requests


class DocumentProcessor:
    """Processes PDF documents and uses LLM to normalize to subject_details JSON structure."""

    def __init__(self, llm_endpoint: str = "http://localhost:1234/v1/chat/completions"):
        self.llm_endpoint = llm_endpoint

    def check_llm_available(self) -> bool:
        """Check if the LLM endpoint is available."""
        try:
            response = requests.get(self.llm_endpoint.replace("/v1/chat/completions", "/v1/models"), timeout=5)
            return response.status_code == 200
        except:
            return False

    def process_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process uploaded PDF file and extract structured subject details using LLM.

        Args:
            file_content: Raw PDF file bytes
            filename: Original filename

        Returns:
            Normalized subject_details JSON
        """
        if not filename.lower().endswith('.pdf'):
            raise ValueError("Only PDF files are supported")

        # Check if LLM is available
        if not self.check_llm_available():
            raise Exception("LLM service not available. Please ensure LM Studio is running with the Qwen model loaded.")

        # Extract text from PDF
        text = self._extract_text_from_pdf(file_content)

        # Use LLM to parse and organize the text
        return self._parse_with_llm(text)

    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF using PyPDF2."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")

    def _parse_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Use LLM to parse and organize PDF text into structured JSON format.
        """
        prompt = f"""You are an expert at analyzing course documents and syllabi.

Please analyze the following course document text and extract the key information into a structured JSON format. Focus on identifying:

1. Module/Course Title
2. Learning Goal/Objective
3. Prerequisites/Requirements
4. Learning Outcomes/Objectives (with Bloom's taxonomy levels)
5. Syllabus/Course Content (organized by chapters/units)

Document Text:
{text[:4000]}  # Limit text length for LLM

Return ONLY valid JSON in this exact format:
{{
  "subject_details": {{
    "module_title": "Course Title Here",
    "learning_goal": "Main learning objective",
    "prerequisites": ["prereq 1", "prereq 2"],
    "learning_outcomes": [
      {{
        "id": "LO1",
        "description": "Learning outcome description",
        "bloom_level": "Understand"
      }}
    ],
    "syllabus": [
      {{
        "chapter": 1,
        "title": "Chapter Title",
        "topics": ["topic 1", "topic 2"]
      }}
    ]
  }}
}}

For Bloom's taxonomy levels, use: Remember, Understand, Apply, Analyze, Evaluate, Create
Be thorough but concise. If information is not available, use empty arrays or empty strings."""

        try:
            response = requests.post(
                self.llm_endpoint,
                json={
                    "model": "qwen/qwen2.5-vl-7b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.3
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']

                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    parsed_json = json.loads(json_match.group())
                    return self._validate_and_clean_json(parsed_json)
                else:
                    raise Exception("No JSON found in LLM response")
            else:
                raise Exception(f"LLM API error: {response.status_code}")

        except Exception as e:
            # Fallback to regex parsing if LLM fails
            print(f"LLM parsing failed: {e}. Using regex fallback.")
            return self._parse_text_to_json(text)

    def _validate_and_clean_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the LLM-generated JSON."""
        if 'subject_details' not in data:
            raise ValueError("Invalid JSON structure: missing subject_details")

        sd = data['subject_details']

        # Ensure required fields exist
        required_fields = ['module_title', 'learning_goal', 'prerequisites', 'learning_outcomes', 'syllabus']
        for field in required_fields:
            if field not in sd:
                sd[field] = [] if field in ['prerequisites', 'learning_outcomes', 'syllabus'] else ""

        # Validate learning outcomes structure
        if isinstance(sd['learning_outcomes'], list):
            for i, outcome in enumerate(sd['learning_outcomes']):
                if isinstance(outcome, dict):
                    if 'id' not in outcome:
                        outcome['id'] = f"LO{i+1}"
                    if 'bloom_level' not in outcome:
                        outcome['bloom_level'] = self._infer_bloom_level(outcome.get('description', ''))
                else:
                    # Convert string to proper structure
                    sd['learning_outcomes'][i] = {
                        "id": f"LO{i+1}",
                        "description": str(outcome),
                        "bloom_level": self._infer_bloom_level(str(outcome))
                    }

        # Validate syllabus structure
        if isinstance(sd['syllabus'], list):
            for chapter in sd['syllabus']:
                if isinstance(chapter, dict):
                    if 'chapter' not in chapter:
                        chapter['chapter'] = ""
                    if 'topics' not in chapter:
                        chapter['topics'] = []
                else:
                    # Try to convert string chapters
                    chapter_match = re.match(r'(?:Chapter|Unit)?\s*(\d+|[IVXLCDM]+)[\:\.\s]*(.+)', str(chapter), re.IGNORECASE)
                    if chapter_match:
                        chapter_num = chapter_match.group(1)
                        title = chapter_match.group(2).strip()
                        sd['syllabus'][sd['syllabus'].index(chapter)] = {
                            "chapter": chapter_num,
                            "title": title,
                            "topics": []
                        }

        return data

    def _parse_text_to_json(self, text: str) -> Dict[str, Any]:
        """
        Fallback regex-based parsing of extracted text into normalized JSON structure.
        """
        # Initialize structure
        result = {
            "subject_details": {
                "module_title": "",
                "learning_goal": "",
                "prerequisites": [],
                "learning_outcomes": [],
                "syllabus": []
            }
        }

        # Clean and normalize text
        text = re.sub(r'\n+', '\n', text.strip())

        # Extract module title (look for common patterns)
        title_patterns = [
            r'Module\s*Title\s*:\s*(.+?)(?:\n|$)',
            r'Course\s*Title\s*:\s*(.+?)(?:\n|$)',
            r'Subject\s*:\s*(.+?)(?:\n|$)',
            r'^(.+?)(?:\n|$)'  # First line as fallback
        ]

        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                result["subject_details"]["module_title"] = match.group(1).strip()
                break

        # Extract learning goal
        goal_patterns = [
            r'Learning\s*Goal\s*:\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'Objective\s*:\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'Goal\s*:\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'General\s*Objective\s*:\s*(.+?)(?:\n\n|\n[A-Z]|$)'
        ]

        for pattern in goal_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["subject_details"]["learning_goal"] = match.group(1).strip()
                break

        # Extract prerequisites
        prereq_section = self._extract_section(text, ['prerequisites', 'prerequisite', 'requirements'])
        if prereq_section:
            # Split by bullets or numbers
            prereqs = re.split(r'[•\-\*\d]+\s*', prereq_section)
            result["subject_details"]["prerequisites"] = [p.strip() for p in prereqs if p.strip()]

        # Extract learning outcomes
        outcomes_section = self._extract_section(text, ['learning outcomes', 'learning objectives', 'outcomes'])
        if outcomes_section:
            # Look for numbered or bulleted items
            outcomes = re.findall(r'[•\-\*\d]+\s*(.+?)(?=[•\-\*\d]|$)', outcomes_section, re.DOTALL)
            for i, outcome in enumerate(outcomes, 1):
                result["subject_details"]["learning_outcomes"].append({
                    "id": f"LO{i}",
                    "description": outcome.strip(),
                    "bloom_level": self._infer_bloom_level(outcome.strip())
                })

        # Extract syllabus
        syllabus_section = self._extract_section(text, ['syllabus', 'course content', 'topics', 'chapters'])
        if syllabus_section:
            chapters = self._parse_syllabus(syllabus_section)
            result["subject_details"]["syllabus"] = chapters

        return result

    def _extract_section(self, text: str, keywords: List[str]) -> str:
        """Extract a section of text based on keywords."""
        lines = text.split('\n')
        section_lines = []
        in_section = False

        for line in lines:
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in keywords):
                in_section = True
                continue
            elif in_section and line.strip() and not line.startswith(' ') and len(section_lines) > 0:
                # Likely moved to next section
                break
            elif in_section:
                section_lines.append(line)

        return '\n'.join(section_lines).strip()

    def _parse_syllabus(self, syllabus_text: str) -> List[Dict[str, Any]]:
        """Parse syllabus text into chapter structure."""
        chapters = []
        lines = syllabus_text.split('\n')

        current_chapter = None
        current_title = ""
        current_topics = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if it's a chapter header
            chapter_match = re.match(r'(?:Chapter|Unit|Module)\s*(\d+|[IVXLCDM]+)[\:\.\s]*(.+)', line, re.IGNORECASE)
            if chapter_match:
                # Save previous chapter
                if current_chapter is not None:
                    chapters.append({
                        "chapter": current_chapter,
                        "title": current_title,
                        "topics": current_topics
                    })

                # Start new chapter
                chapter_num = chapter_match.group(1)
                current_title = chapter_match.group(2).strip()
                try:
                    current_chapter = int(chapter_num)
                except:
                    current_chapter = chapter_num
                current_topics = []
            elif current_chapter is not None and (line.startswith('•') or line.startswith('-') or re.match(r'\d+\.', line)):
                # It's a topic
                topic = re.sub(r'^[•\-\*\d\.\s]+', '', line).strip()
                if topic:
                    current_topics.append(topic)

        # Add last chapter
        if current_chapter is not None:
            chapters.append({
                "chapter": current_chapter,
                "title": current_title,
                "topics": current_topics
            })

        return chapters

    def _infer_bloom_level(self, outcome: str) -> str:
        """Infer Bloom's taxonomy level from outcome description."""
        outcome_lower = outcome.lower()

        if any(word in outcome_lower for word in ['create', 'design', 'develop', 'construct']):
            return "Create"
        elif any(word in outcome_lower for word in ['evaluate', 'assess', 'judge', 'critique']):
            return "Evaluate"
        elif any(word in outcome_lower for word in ['analyze', 'compare', 'contrast', 'classify']):
            return "Analyze"
        elif any(word in outcome_lower for word in ['apply', 'use', 'implement', 'demonstrate']):
            return "Apply"
        elif any(word in outcome_lower for word in ['understand', 'explain', 'describe', 'discuss']):
            return "Understand"
        else:
            return "Remember"

    def _normalize_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize existing JSON to our structure."""
        # If already in our format, return as-is
        if 'subject_details' in data:
            return data

        # Otherwise, try to map common fields
        normalized = {
            "subject_details": {
                "module_title": "",
                "learning_goal": "",
                "prerequisites": [],
                "learning_outcomes": [],
                "syllabus": []
            }
        }

        # Map module title
        normalized["subject_details"]["module_title"] = data.get('title', data.get('course_title', data.get('subject', data.get('module_title', ''))))

        # Map learning goal
        normalized["subject_details"]["learning_goal"] = data.get('goal', data.get('objective', data.get('learning_goal', data.get('general_objective', ''))))

        # Map prerequisites
        normalized["subject_details"]["prerequisites"] = data.get('prerequisites', data.get('requirements', []))

        # Map learning outcomes
        if 'learning_outcomes' in data:
            for i, outcome in enumerate(data['learning_outcomes'], 1):
                if isinstance(outcome, str):
                    normalized["subject_details"]["learning_outcomes"].append({
                        "id": f"LO{i}",
                        "description": outcome,
                        "bloom_level": self._infer_bloom_level(outcome)
                    })
                elif isinstance(outcome, dict):
                    normalized["subject_details"]["learning_outcomes"].append({
                        "id": outcome.get('id', f"LO{i}"),
                        "description": outcome.get('description', ''),
                        "bloom_level": outcome.get('bloom_level', self._infer_bloom_level(outcome.get('description', '')))
                    })

        # Map syllabus
        if 'syllabus' in data:
            for chapter in data['syllabus']:
                if isinstance(chapter, dict):
                    normalized["subject_details"]["syllabus"].append({
                        "chapter": chapter.get('chapter'),
                        "title": chapter.get('title', ''),
                        "topics": chapter.get('topics', [])
                    })

        return normalized
