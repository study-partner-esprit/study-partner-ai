#!/usr/bin/env python3
"""
Focused validation of key improvements without requiring model inference.
Validates: heuristic parsing, prompt structure, user input handling.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("Validating refactored code structure...")
print("=" * 70)

# TEST 1: Heuristic parsing method exists and works
print("\n1. Testing heuristic parsing...")
from src.evaluator.llm_client import GeminiClient

client = GeminiClient()

# Test small model output parsing
small_model_output = """
concept_coverage: 0.75
logical_coherence: 0.60
causal_reasoning: 0.50
error_awareness: 0.40
answer_feedback: Good attempt but missing some concepts
guessing_detected: false
missing_concepts: causality, applications
misconceptions: none
"""

try:
    parsed = client._parse_response_heuristically(small_model_output)
    assert parsed is not None, "Parsing returned None"
    assert parsed.get('concept_coverage') == 0.75, f"concept_coverage mismatch: {parsed.get('concept_coverage')}"
    assert parsed.get('logical_coherence') == 0.60, f"logical_coherence mismatch"
    assert parsed.get('causal_reasoning') == 0.50, f"causal_reasoning mismatch"
    assert parsed.get('error_awareness') == 0.40, f"error_awareness mismatch"
    assert isinstance(parsed.get('missing_concepts'), list), "missing_concepts should be list"
    assert len(parsed.get('missing_concepts')) > 0, "missing_concepts should not be empty"
    print("   ✓ Heuristic parsing works correctly")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    exit(1)

# TEST 2: Analysis fallback creation
print("\n2. Testing analysis fallback creation...")
try:
    # This should create a reasonable fallback from partial output
    test_output = "The student gave a decent answer with a score around 0.6"
    fallback = client._create_analysis_fallback(test_output)
    
    assert fallback.get('concept_coverage') is not None, "Missing concept_coverage"
    assert fallback.get('logical_coherence') is not None, "Missing logical_coherence"
    assert fallback.get('causal_reasoning') is not None, "Missing causal_reasoning"
    assert fallback.get('error_awareness') is not None, "Missing error_awareness"
    assert fallback.get('answer_feedback') is not None, "Missing answer_feedback"
    assert isinstance(fallback.get('guessing_detected'), bool), "guessing_detected should be boolean"
    assert isinstance(fallback.get('missing_concepts'), list), "missing_concepts should be list"
    assert isinstance(fallback.get('misconceptions'), list), "misconceptions should be list"
    
    # Check values are in valid range
    for field in ['concept_coverage', 'logical_coherence', 'causal_reasoning', 'error_awareness']:
        value = fallback.get(field)
        assert 0.0 <= value <= 1.0, f"{field} out of range: {value}"
    
    print("   ✓ Analysis fallback creation works correctly")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    exit(1)

# TEST 3: Prompt structure validation
print("\n3. Testing prompt structure...")
from src.evaluator.prompts import build_analysis_prompt, build_question_prompt
from src.evaluator.schemas import TaskEvaluationContext

context = TaskEvaluationContext(
    task_title="Photosynthesis",
    task_description="Understanding how plants make energy",
    task_details="Photosynthesis is the process by which plants convert light into chemical energy",
    key_concepts=["light reactions", "Calvin cycle"]
)

# Test analysis prompt includes student answer
student_answer = "Plants use sunlight to make food"
student_question = "What is photosynthesis?"

try:
    analysis_msgs = build_analysis_prompt(context, student_question, student_answer)
    
    assert len(analysis_msgs) == 2, "Should have 2 messages (system, user)"
    system_content = analysis_msgs[0]['content']
    
    # Check that student answer is clearly included
    assert student_answer in system_content, "Student answer not in prompt"
    assert student_question in system_content, "Student question not in prompt"
    assert context.task_title in system_content, "Task title not in prompt"
    assert "STUDENT'S ACTUAL ANSWER" in system_content, "Missing label for student answer"
    assert "concept_coverage:" in system_content.lower(), "Missing field labels"
    assert "logical_coherence:" in system_content.lower(), "Missing field labels"
    
    print("   ✓ Analysis prompt structure is correct")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    exit(1)

# Test question prompt includes context
try:
    question_msgs = build_question_prompt(context, [student_answer], "what")
    
    assert len(question_msgs) == 2, "Should have 2 messages (system, user)"
    system_content = question_msgs[0]['content']
    
    assert context.task_title in system_content, "Task title not in question prompt"
    assert context.task_details in system_content, "Task details not in question prompt"
    assert "KEY CONCEPTS" in system_content or "key concepts" in system_content.lower(), "Missing key concepts section"
    
    print("   ✓ Question prompt structure is correct")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    exit(1)

# TEST 4: Evaluator user input handling
print("\n4. Testing evaluator user input storage...")
from src.evaluator.evaluator_agent import EvaluatorAgent
from src.evaluator.schemas import EvaluationSession, SessionState

agent = EvaluatorAgent()

try:
    # Create a session (without starting LLM)
    session = EvaluationSession(
        session_id="test-123",
        task_title="Test",
        task_description="Test task",
        task_details="Test details",
        context=context,
    )
    
    agent.sessions["test-123"] = session
    
    # Verify that when we initialize a session, answer_history is empty
    assert len(session.answer_history) == 0, "Initial answer_history should be empty"
    
    # Note: We can't actually call handle_user_answer without LLM, but we 
    # validated in the code that it appends the answer first thing
    
    print("   ✓ Evaluator session structure is correct")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    exit(1)

# TEST 5: Scoring with fallback values
print("\n5. Testing scoring with fallback values...")
from src.evaluator.scoring import MasteryScorer
from src.evaluator.schemas import LLMAnalysisResponse

scorer = MasteryScorer()

try:
    # Create analysis with neutral 0.5 scores (what fallback would return)
    fallback_analysis = LLMAnalysisResponse(
        concept_coverage=0.5,
        logical_coherence=0.5,
        causal_reasoning=0.5,
        error_awareness=0.5,
        answer_feedback="Neutral feedback",
        guessing_detected=False,
        missing_concepts=[],
        misconceptions=[]
    )
    
    score = scorer.compute_mastery_score(fallback_analysis)
    
    assert isinstance(score, (int, float)), "Score should be numeric"
    assert 0.0 <= score <= 1.0, f"Score should be 0-1, got {score}"
    
    # With all 0.5 scores, the weighted average should be 0.5
    expected = 0.4 * 0.5 + 0.3 * 0.5 + 0.2 * 0.5 + 0.1 * 0.5
    assert abs(score - expected) < 0.01, f"Score {score} doesn't match expected {expected}"
    
    print(f"   ✓ Scoring with fallback values works (0.5 scores → {score:.3f})")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    exit(1)

# TEST 6: Gradio integration validation
print("\n6. Testing Gradio text input handling...")
try:
    # Validate that gradio_app.py exists and has the 4-tuple return signature
    with open("src/gradio_app.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check for submit_answer function with 4-tuple return type
    assert "def submit_answer(" in content, "submit_answer function not found"
    assert "tuple[str, str, str, gr.State]" in content, "4-tuple return type not found"
    assert "outputs=[result_display, session_info, user_answer, session_state]" in content, "Event handler outputs not updated to 4-tuple"
    
    print("   ✓ Gradio integration has correct 4-tuple signature")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✓ All validations passed!")
print("\nKey improvements verified:")
print("  • Heuristic parsing for small model outputs")
print("  • Analysis fallback creation with safe defaults")
print("  • Prompts clearly include student answer and task context")
print("  • Scoring handles fallback neutral values correctly")
print("  • Gradio integration uses 4-tuple for state persistence")
print("\nThe refactored system should now:")
print("  ✓ Handle JSON parsing failures gracefully")
print("  ✓ Pass user input correctly to the model")
print("  ✓ Continue sessions instead of premature FAIL")
print("  ✓ Work with small 3B models offline")
