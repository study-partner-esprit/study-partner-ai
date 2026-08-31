"""
Prompt templates for Gemini-based Socratic evaluation.
Generates questions and analyzes answers in plain text.
Includes template-based local question generation to minimize API calls.
"""

import random
import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def extract_score(text: str) -> float:
    """
    Robustly extract score from LLM response text.
    Handles malformed responses like "Score: 0." or missing decimal.
    Strips whitespace, uses strict regex, clamps to valid range.

    Args:
        text: Text containing score in format "Score: 0.XX"

    Returns:
        Score as float in [0.0, 1.0], or None if parsing completely fails
    """
    if not text or not isinstance(text, str):
        return None

    # Strip all whitespace and normalize
    text = re.sub(r'\s+', '', text).lower()

    # Strict regex: match 0.0-1.0 with optional decimal places
    # Examples: "0.85", "1.0", "0.5", "0", "1"
    match = re.search(r'(?:^|[^0-9])(0(?:\.\d+)?|1(?:\.0+)?)(?:[^0-9]|$)', text)

    if match:
        score_text = match.group(1)

        # Reject incomplete patterns like "0." or "."
        if score_text.endswith('.') or score_text == '.':
            logger.warning(f"Invalid LLM score - incomplete decimal: '{score_text}'")
            return None

        try:
            score = float(score_text)
            # Clamp to valid range
            score = max(0.0, min(1.0, score))
            logger.debug(f"Successfully extracted LLM score: {score}")
            return score
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to convert score '{score_text}' to float: {e}")
            return None

    logger.warning(f"No valid score pattern found in text: '{text[:100]}'")
    return None


def parse_analysis_response(text: str) -> dict:
    """
    Parse structured analysis response from Gemini.
    Extracts: Score, Strengths, Weaknesses, Missing Concepts
    Returns robust defaults if parsing fails.

    Args:
        text: Gemini response in structured format

    Returns:
        Dict with keys: score, strengths, weaknesses, missing_concepts
        score may be None if parsing fails completely
    """
    result = {
        "score": None,  # Changed: None instead of 0.5 default
        "strengths": "",
        "weaknesses": "",
        "missing_concepts": []
    }

    if not text or not isinstance(text, str):
        logger.warning("Empty or invalid analysis text provided")
        return result

    # Extract score using improved extraction
    result["score"] = extract_score(text)

    # Extract strengths section
    strengths_match = re.search(r"Strengths:\s*(.+?)(?=Weaknesses:|$)", text, re.DOTALL | re.IGNORECASE)
    if strengths_match:
        result["strengths"] = strengths_match.group(1).strip()

    # Extract weaknesses section
    weaknesses_match = re.search(r"Weaknesses:\s*(.+?)(?=Missing|$)", text, re.DOTALL | re.IGNORECASE)
    if weaknesses_match:
        result["weaknesses"] = weaknesses_match.group(1).strip()

    # Extract missing concepts section
    missing_match = re.search(r"Missing Concepts:\s*(.+?)$", text, re.DOTALL | re.IGNORECASE)
    if missing_match:
        concepts_text = missing_match.group(1).strip()
        # Split by comma or newline, filter empty items
        concepts = re.split(r'[,\n]', concepts_text)
        result["missing_concepts"] = [c.strip() for c in concepts if c.strip()]

    # Fall back to sentence parsing when structured sections are missing
    if not result["strengths"] and not result["weaknesses"]:
        normalized = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', normalized)
        if sentences:
            result["strengths"] = sentences[0][:250]
        if len(sentences) > 1:
            result["weaknesses"] = sentences[1][:250]

    return result


def clean_concepts(concepts: list[str]) -> list[str]:
    """Filter concepts by quality: remove banned words, prefer length >= 5."""
    banned = {
        "process", "thing", "system", "concept", "stuff",
        "using", "making", "having", "doing", "important",
        "different", "other", "parts", "happens", "occurs",
        "takes", "place", "works", "functions", "operates",
        "involves", "requires", "needs", "helps", "allows",
        "provides", "creates", "produces", "generates", "causes",
        "results", "leads", "brings", "gives", "makes", "gets",
        "comes", "goes", "starts", "begins", "ends", "finishes",
        "continues", "stops", "changes", "becomes", "remains",
        "stays", "keeps", "holds", "maintains", "supports",
        "enables", "facilitates", "promotes", "enhances", "improves",
        "increases", "decreases", "reduces", "raises", "lowers",
        "moves", "flows", "passes", "enters", "exits", "leaves",
        "joins", "connects", "links", "binds", "attaches", "separates",
        "divides", "splits", "breaks", "forms", "shapes", "builds",
        "constructs", "develops", "grows", "expands", "shrinks",
        "contracts", "transforms", "converts", "changes", "alters",
        "modifies", "adjusts", "controls", "manages", "handles",
        "directs", "guides", "leads", "follows", "tracks", "monitors",
        "watches", "observes", "checks", "tests", "measures",
        "counts", "calculates", "computes", "determines", "finds",
        "discovers", "identifies", "recognizes", "knows", "understands",
        "learns", "studies", "examines", "analyzes", "evaluates",
        "assesses", "judges", "decides", "chooses", "selects",
        "picks", "takes", "uses", "applies", "employs", "utilizes",
        "consumes", "absorbs", "releases", "emits", "radiates",
        "transmits", "receives", "sends", "delivers", "transports",
        "carries", "moves", "shifts", "pushes", "pulls", "drives",
        "propels", "attracts", "repels", "combines", "mixes",
        "blends", "merges", "unites", "separates", "divides"
    }
    return [
        c.strip()
        for c in concepts
        if c and c.strip().lower() not in banned and len(c.strip()) >= 5
    ]


def has_generic_question_terms(question: str) -> bool:
    """Detect generic words that weaken Socratic questions."""
    if not question or not isinstance(question, str):
        return False
    generic_words = {
        "process", "system", "thing", "concept", "stuff",
        "item", "part", "component", "aspect", "element"
    }
    pattern = r"\b(" + "|".join(re.escape(word) for word in generic_words) + r")\b"
    return bool(re.search(pattern, question.lower()))


def question_contains_concept(question: str, concepts: list[str]) -> bool:
    """Check if the question mentions at least one cleaned key concept."""
    if not question:
        return False
    if not concepts:
        return True
    question_lower = question.lower()
    for concept in clean_concepts(concepts):
        if concept.lower() in question_lower:
            return True
    return False


def extract_keywords_from_text(text: str) -> list[str]:
    """Extract and rank keywords by frequency, return top 5 most relevant."""
    if not text:
        return []

    words = re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())  # Prefer >= 5 chars
    banned = {
        "process", "thing", "system", "concept", "stuff",
        "using", "making", "having", "doing", "important",
        "different", "other", "parts", "happens", "occurs",
        "takes", "place", "works", "functions", "operates",
        "involves", "requires", "needs", "helps", "allows",
        "provides", "creates", "produces", "generates", "causes",
        "results", "leads", "brings", "gives", "makes", "gets",
        "comes", "goes", "starts", "begins", "ends", "finishes",
        "continues", "stops", "changes", "becomes", "remains",
        "stays", "keeps", "holds", "maintains", "supports",
        "enables", "facilitates", "promotes", "enhances", "improves",
        "increases", "decreases", "reduces", "raises", "lowers",
        "moves", "flows", "passes", "enters", "exits", "leaves",
        "joins", "connects", "links", "binds", "attaches", "separates",
        "divides", "splits", "breaks", "forms", "shapes", "builds",
        "constructs", "develops", "grows", "expands", "shrinks",
        "contracts", "transforms", "converts", "changes", "alters",
        "modifies", "adjusts", "controls", "manages", "handles",
        "directs", "guides", "leads", "follows", "tracks", "monitors",
        "watches", "observes", "checks", "tests", "measures",
        "counts", "calculates", "computes", "determines", "finds",
        "discovers", "identifies", "recognizes", "knows", "understands",
        "learns", "studies", "examines", "analyzes", "evaluates",
        "assesses", "judges", "decides", "chooses", "selects",
        "picks", "takes", "uses", "applies", "employs", "utilizes",
        "consumes", "absorbs", "releases", "emits", "radiates",
        "transmits", "receives", "sends", "delivers", "transports",
        "carries", "moves", "shifts", "pushes", "pulls", "drives",
        "propels", "attracts", "repels", "combines", "mixes",
        "blends", "merges", "unites", "separates", "divides"
    }
    stop_words = {
        "the", "that", "this", "these", "those", "with", "from",
        "then", "when", "they", "their", "there", "about", "which",
        "would", "could", "should", "your", "have", "what", "where",
        "while", "under", "after", "before", "within", "should", "being",
        "during", "through", "because", "however", "therefore", "thus",
        "hence", "consequently", "accordingly", "moreover", "furthermore",
        "additionally", "similarly", "likewise", "also", "besides",
        "further", "more", "again", "still", "yet", "even", "though",
        "although", "despite", "notwithstanding", "whereas", "while",
        "whereby", "wherein", "whereupon", "wherever", "whether",
        "whenever", "wherever", "whichever", "whatever", "whoever",
        "whomever", "whose", "how", "why", "when", "where", "what",
        "which", "who", "whom", "whose", "how", "why", "when", "where"
    }

    cleaned = [w for w in words if w not in banned and w not in stop_words]
    word_freq = Counter(cleaned)
    
    # Prefer words that appear multiple times, then by length
    candidates = [(word, freq, len(word)) for word, freq in word_freq.items()]
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)  # freq desc, then length desc
    
    top_words = [word for word, _, _ in candidates[:5]]
    return top_words


def validate_question(question: str) -> str:
    """Validate question: ends with ?, at least 8 words, no duplicates."""
    question = re.sub(r"\s+", " ", question).strip()

    if not question:
        logger.warning("Rejected empty question during validation")
        return "Can you explain how this works step by step and why each stage matters?"

    if not question.endswith("?"):
        question = f"{question}?"

    word_count = len(question.split())
    if word_count < 8:
        logger.warning(
            "Rejected question during validation: too few words (%d): %s",
            word_count,
            question,
        )
        return "Can you explain how this works step by step and why each stage matters?"

    # Remove duplicate consecutive words (e.g., "process process" → "process")
    words = question.split()
    deduped = []
    for word in words:
        if not deduped or word.lower() != deduped[-1].lower():
            deduped.append(word)

    validated = " ".join(deduped).strip()
    logger.debug("Validated question: %s", validated)
    return validated


def concept_coverage(answer: str, concepts: list[str]) -> float:
    """
    Compute accurate concept coverage score (0.0-1.0).
    Only counts meaningful scientific/domain concepts that appear in the answer.

    Args:
        answer: Student's text answer
        concepts: List of key concepts to check for

    Returns:
        Coverage score: (# covered concepts / total concepts), or 0.0 if no concepts
    """
    if not concepts or not answer:
        return 0.0

    # Clean the concepts first (remove banned/generic words)
    valid_concepts = clean_concepts(concepts)
    if not valid_concepts:
        return 0.0

    answer_lower = answer.lower()
    covered_count = 0

    for concept in valid_concepts:
        concept_lower = concept.lower()
        # Check for exact concept match (case-insensitive)
        if concept_lower in answer_lower:
            covered_count += 1
            logger.debug(f"Concept covered: '{concept}'")
        else:
            logger.debug(f"Concept missing: '{concept}'")

    coverage = covered_count / len(valid_concepts)
    logger.debug(f"Concept coverage: {covered_count}/{len(valid_concepts)} = {coverage:.3f}")

    return coverage


def generate_template_question(
    depth_level: str,
    key_concepts: list[str],
    task_title: str,
    task_details: str = "",
    attempt_number: int = 1,
    used_concepts: set = None
) -> str:
    """
    Generate a question using depth-level templates with key concepts.
    Avoids repetition by tracking used concepts across attempts.
    Always validates concepts and uses safe fallbacks.
    
    Args:
        depth_level: "what" (definition), "why" (reasoning), "how" (mechanism)
        key_concepts: List of key concepts for the task
        task_title: Title of the task for context
        task_details: Detailed task description
        attempt_number: Attempt number for template rotation
        used_concepts: Set of concepts already used (to avoid repetition)
    
    Returns:
        A complete, validated question with meaningful concepts or safe fallback
    """
    if used_concepts is None:
        used_concepts = set()
    
    # Clean and validate input concepts
    valid_concepts = clean_concepts(key_concepts)
    valid_concepts = [c for c in valid_concepts if c.lower().strip() != task_title.lower().strip()]
    
    # Avoid reusing concepts
    available_concepts = [c for c in valid_concepts if c.lower() not in {u.lower() for u in used_concepts}]
    if not available_concepts:
        available_concepts = valid_concepts
    
    selected_concept = None
    if available_concepts:
        selected_concept = random.choice(available_concepts)
    
    # Fallback extraction from task_details with additional validation
    if not selected_concept:
        fallback_concepts = extract_keywords_from_text(task_details)
        fallback_concepts = [c for c in fallback_concepts if c.lower().strip() != task_title.lower().strip()]
        # Additional validation: ensure fallback concepts are meaningful
        fallback_concepts = [c for c in fallback_concepts if len(c) >= 5 and c.isalpha()]
        if fallback_concepts:
            selected_concept = random.choice(fallback_concepts)
    
    # Final validation: concept must be meaningful
    concept = None
    if selected_concept and selected_concept.strip():
        # Double-check it's not banned and meaningful
        test_clean = clean_concepts([selected_concept])
        if test_clean and len(test_clean[0]) >= 5:
            concept = test_clean[0]
    
    # Templates organized by depth level
    templates_by_depth = {
        "what": [
            "What is {concept} and why is it significant in this context?",
            "What are the key characteristics of {concept} that define it?",
            "What is the relationship between {concept} and the overall goal?",
            "What specific aspects of {concept} should we understand first?",
            "What distinguishes {concept} from related ideas?",
            "What role does {concept} play in this learning task?",
        ],
        "why": [
            "Why is {concept} important in this process, and what would happen without it?",
            "Why does {concept} matter for achieving the learning objectives?",
            "Why is {concept} necessary, and how does it support the outcome?",
            "Why do you think {concept} works this way in the system?",
            "Why is {concept} considered a key part of understanding this topic?",
            "Why does {concept} influence other parts of the system?",
        ],
        "how": [
            "How does {concept} interact with other components in this process?",
            "How does {concept} contribute to the overall system functioning?",
            "How would you apply your understanding of {concept} in practice?",
            "How does {concept} connect to the broader process or system?",
            "How do changes in {concept} affect the overall outcome?",
            "How would you demonstrate your understanding of {concept}?",
        ]
    }
    
    # Safe fallback templates that reference task_title
    fallback_templates = [
        f"Can you explain the main ideas of {task_title} and why they matter?",
        f"What specific example best illustrates how {task_title} works in this context?",
        f"Why is understanding {task_title} important in the broader field?",
        f"How would you use {task_title} to solve a real problem?",
        f"What makes {task_title} essential for understanding this topic?",
    ]
    
    # Select template based on depth level
    depth = depth_level.lower() if depth_level else "what"
    if depth not in templates_by_depth:
        depth = "what"
    
    if concept:
        templates = templates_by_depth[depth].copy()
        other_concepts = [c for c in valid_concepts if c.lower() != concept.lower()]
        reference_concept = other_concepts[0] if other_concepts else "the overall system"
    else:
        templates = fallback_templates
        reference_concept = "the overall system"
    
    # Filter out templates with concept placeholders if concept is None
    safe_templates = []
    for template in templates:
        if "{concept}" in template and not concept:
            continue
        safe_templates.append(template)
    
    if not safe_templates:
        safe_templates = fallback_templates

    def build_candidate(template: str) -> str:
        candidate = template.format(concept=concept, reference=reference_concept) if concept else template
        candidate = validate_question(candidate)
        if not candidate:
            return ""
        if has_generic_question_terms(candidate):
            return ""
        if concept and not question_contains_concept(candidate, [concept]):
            return ""
        return candidate

    question = ""
    for _ in range(5):
        candidate = build_candidate(random.choice(safe_templates))
        if candidate:
            question = candidate
            break

    if not question:
        if concept:
            question = validate_question(
                f"How does {concept} influence the key ideas in {task_title} and why does it matter?"
            )
        else:
            question = validate_question(
                f"How does {task_title} work in a meaningful, concept-specific way?"
            )

    return question


def generate_followup_question(
    depth_level: str,
    key_concepts: list[str],
    student_answer: str,
    attempt_number: int
) -> str:
    """
    Generate follow-up questions based on depth level (no API call).
    Used for attempts 2+ to minimize API usage.
    Includes adaptive hints for higher attempt numbers.
    
    Args:
        depth_level: "what" (definition), "why" (reasoning), "how" (mechanism)
        key_concepts: List of key concepts
        student_answer: Student's previous answer
        attempt_number: Current attempt number (2+)
    
    Returns:
        A validated follow-up question with optional hints
    """
    
    valid_concepts = clean_concepts(key_concepts)
    if not valid_concepts:
        fallback = "Can you provide more detail and give a specific example?"
        return validate_question(fallback)

    selected_concept = random.choice(valid_concepts)
    other_concepts = [c for c in valid_concepts if c.lower() != selected_concept.lower()]
    related_reference = other_concepts[0] if other_concepts else "other components"

    # Follow-up templates organized by depth level
    templates = {
        "what": [
            f"You mentioned some good points. What specifically does {selected_concept} mean in this context?",
            f"Your answer covers part of it. What else should we know about {selected_concept}?",
            f"That's helpful. What is the most critical characteristic of {selected_concept}?",
            f"You've started well. What specific aspects of {selected_concept} did you notice?",
            f"Can you clarify what {selected_concept} includes and what it excludes?",
        ],
        "why": [
            f"You've explained what happens. Now, why do you think {selected_concept} works this way?",
            f"That's a good observation. But why is {selected_concept} important here?",
            f"You've identified part of it. Why does {selected_concept} matter for the goal?",
            f"Can you dig deeper? Why is {selected_concept} necessary for this process?",
            f"You've partially answered. Why does {selected_concept} interact with {related_reference} this way?",
        ],
        "how": [
            f"Great start. How would you apply your understanding of {selected_concept} in practice?",
            f"You understand the concept. How does {selected_concept} actually work step by step?",
            f"Good thinking. How does {selected_concept} connect to the overall process?",
            f"You're on the right track. How would {selected_concept} interact with {related_reference}?",
            f"Excellent insight. How could understanding {selected_concept} help solve a real problem?",
        ]
    }
    
    depth = depth_level.lower() if depth_level else "what"
    if depth not in templates:
        depth = "what"
    
    question_set = templates[depth]
    template_index = (attempt_number - 2) % len(question_set)
    question = question_set[template_index]
    
    # Add adaptive hints for attempts > 2
    if attempt_number > 2:
        hints = [
            " Try thinking about specific examples or applications.",
            " Consider how this relates to real-world scenarios.",
            " Think about the step-by-step process involved.",
            " Try to connect this to what you already know.",
        ]
        hint_index = (attempt_number - 2) % len(hints)
        question += hints[hint_index]

    question = validate_question(question)
    if has_generic_question_terms(question) or not question_contains_concept(question, valid_concepts):
        fallback = f"How does {selected_concept} specifically contribute to understanding this task?"
        return validate_question(fallback)

    return question


def build_question_prompt(
    task_title: str,
    task_description: str,
    task_details: str,
    key_concepts: list[str],
    depth_level: str = "what"
) -> str:
    """
    Build Socratic question prompt for Gemini.
    Specifies depth level to guide question generation.
    
    Args:
        task_title: Title of the learning task
        task_description: Brief description
        task_details: Detailed task information
        key_concepts: List of concepts to evaluate
        depth_level: "what" (definition), "why" (reasoning), "how" (mechanism)
    
    Returns:
        Prompt text for Gemini
    """
    
    depth_guidance = {
        "what": "Focus on DEFINITION and KEY CHARACTERISTICS. Ask about what something IS or means.",
        "why": "Focus on REASONING and IMPORTANCE. Ask about WHY something matters or happens.",
        "how": "Focus on MECHANISM and APPLICATION. Ask about HOW something works or is used."
    }
    
    guidance = depth_guidance.get(depth_level.lower(), depth_guidance["what"])
    
    prompt = f'''You are a Socratic tutor assessing understanding.

Task: {task_title}
Description: {task_description}

Details:
{task_details}

Key concepts: {', '.join(key_concepts)}

DEPTH LEVEL: {depth_level.upper()}
{guidance}

Generate EXACTLY ONE question to test understanding at this level.
The question MUST:
- Be a complete, natural sentence (NOT a fragment)
- Be at least 12 words long
- End with a question mark
- Focus on the depth level specified above
- Use one key concept at a time
- Avoid generic words like process, system, thing, concept, or stuff
- Do not use vague placeholders; make the question concept-specific
- NOT start mid-sentence

Return ONLY the final question, nothing else.
'''

    return prompt


def build_analysis_prompt(
    task_title: str,
    task_description: str,
    task_details: str,
    key_concepts: list[str],
    student_answer: str,
    previous_answers: list[str] = None
) -> str:
    """
    Build answer analysis prompt for Gemini with structured output.
    Returns assessment in fixed format for consistency.
    
    Args:
        task_title: Title of the learning task
        task_description: Brief description
        task_details: Detailed task information
        key_concepts: List of concepts to evaluate
        student_answer: Current student answer
        previous_answers: Prior answers (optional)
    
    Returns:
        Prompt text with structured output format
    """
    
    if previous_answers is None:
        previous_answers = []
    
    previous_context = ""
    if previous_answers and len(previous_answers) > 1:
        prev_ans = previous_answers[-2]
        previous_context = f"\nPREVIOUS ATTEMPT: {prev_ans[:100]}...\n"
    
    short_description = task_description[:50] if len(task_description) > 50 else task_description
    concepts_str = ", ".join(key_concepts[:3])
    
    prompt = f"""Evaluate understanding of: {task_title} ({short_description})

KEY CONCEPTS TO ASSESS: {concepts_str}
{previous_context}
STUDENT ANSWER:
{student_answer}

Provide structured feedback with NO preamble:

Score: [0.0-1.0 decimal]
Strengths: [1-2 sentences on what they understand correctly]
Weaknesses: [1-2 sentences on gaps or misconceptions]
Missing Concepts: [list specific scientific/domain concepts not demonstrated, max 5, avoid generic words like "process" or "system"]

Be specific and focus on meaningful concepts. High score only if they demonstrate clear understanding of most key concepts."""
    
    return prompt
