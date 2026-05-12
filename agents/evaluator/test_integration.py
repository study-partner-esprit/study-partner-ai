#!/usr/bin/env python3
"""
Quick integration test to verify the refactored evaluator handles small model outputs.
Tests JSON fallback, user input passing, and fallback scoring.
"""

import sys
import logging

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

from src.evaluator.llm_client import GeminiClient
from src.evaluator.evaluator_agent import EvaluatorAgent
from src.evaluator.schemas import LLMAnalysisResponse, LLMQuestionResponse

def test_heuristic_parsing():
    """Test that heuristic parsing works for malformed model outputs."""
    print("\n" + "="*60)
    print("TEST 1: Heuristic Parsing (Model Output)")
    print("="*60)
    
    client = GeminiClient()
    
    # Simulate a small model output that's not valid JSON
    small_model_output = """
    concept_coverage: 0.75
    logical_coherence: 0.60
    causal_reasoning: 0.50
    error_awareness: 0.40
    answer_feedback: The student showed good understanding but missed some deeper concepts.
    guessing_detected: false
    missing_concepts: causal relationships, advanced applications
    misconceptions: none
    """
    
    try:
        parsed = client._parse_response_heuristically(small_model_output)
        print(f"✓ Heuristic parsing succeeded!")
        print(f"  Parsed: {parsed}")
        return True
    except Exception as e:
        print(f"✗ Heuristic parsing failed: {e}")
        return False

def test_evaluator_session():
    """Test that evaluator handles a complete session with fallback scoring."""
    print("\n" + "="*60)
    print("TEST 2: Evaluator Session with Fallback Scoring")
    print("="*60)
    
    try:
        agent = EvaluatorAgent()
        
        # Start a session
        print("\nStarting evaluation session...")
        session_result = agent.start_session(
            task_title="Photosynthesis",
            task_description="Understanding how plants create energy from sunlight",
            task_details="Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in chloroplasts and involves two main stages: light reactions and the Calvin cycle.",
            max_attempts=5
        )
        
        session_id = session_result['session_id']
        first_question = session_result['question']
        print(f"✓ Session started: {session_id}")
        print(f"  First question: {first_question[:80]}...")
        
        # Submit an answer
        user_answer = "Photosynthesis is when plants use sunlight to make food."
        print(f"\nSubmitting answer: '{user_answer[:60]}...'")
        
        result = agent.handle_user_answer(session_id, user_answer)
        
        print(f"✓ Answer processed!")
        print(f"  State: {result.get('state', 'UNKNOWN')}")
        print(f"  Mastery Score: {result.get('mastery_score', 'N/A'):.3f}")
        print(f"  Feedback: {result.get('feedback', 'N/A')[:60]}...")
        
        # Check that answer was recorded
        session = agent.get_session(session_id)
        if user_answer in session.answer_history:
            print(f"✓ User answer was recorded in history")
        else:
            print(f"✗ User answer was NOT recorded in history")
            return False
        
        # Check that we got a fallback score (0.5) if LLM failed
        if 0.0 <= result.get('mastery_score', -1) <= 1.0:
            print(f"✓ Mastery score is valid (0.0-1.0)")
        else:
            print(f"✗ Mastery score is invalid: {result.get('mastery_score')}")
            return False
        
        # Check that session continued (not marked FAILED)
        if result.get('state') in ['CONTINUE', 'MASTERY_CONFIRMED']:
            print(f"✓ Session continued (not marked FAILED)")
            return True
        elif result.get('state') == 'FAILED':
            print(f"✗ Session marked FAILED (should continue): {result.get('message')}")
            return False
        else:
            print(f"? Unknown state: {result.get('state')}")
            return False
            
    except Exception as e:
        print(f"✗ Session test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_formats():
    """Test that prompts include task context and user answer."""
    print("\n" + "="*60)
    print("TEST 3: Prompt Format Validation")
    print("="*60)
    
    from src.evaluator.prompts import build_analysis_prompt, build_question_prompt
    from src.evaluator.schemas import TaskEvaluationContext
    
    context = TaskEvaluationContext(
        task_title="Photosynthesis",
        task_description="Understanding plant energy production",
        task_details="Photosynthesis converts light into chemical energy",
        key_concepts=["light reactions", "Calvin cycle", "chloroplasts"]
    )
    
    # Test analysis prompt
    student_answer = "Plants use sunlight to make food through photosynthesis"
    analysis_prompt = build_analysis_prompt(
        context=context,
        student_question="What is photosynthesis?",
        student_answer=student_answer
    )
    
    # Check that student answer is in the prompt
    system_prompt_text = analysis_prompt[0]['content']
    
    if student_answer in system_prompt_text:
        print(f"✓ Student answer is included in analysis prompt")
    else:
        print(f"✗ Student answer is NOT in analysis prompt")
        return False
    
    if "STUDENT'S ACTUAL ANSWER" in system_prompt_text:
        print(f"✓ Prompt clearly labels the student answer")
    else:
        print(f"✗ Prompt does not clearly label student answer")
        return False
    
    # Test question prompt
    question_prompt = build_question_prompt(
        context=context,
        student_history=[student_answer],
        depth_level="what"
    )
    
    system_prompt_text = question_prompt[0]['content']
    if context.task_title in system_prompt_text and context.task_details in system_prompt_text:
        print(f"✓ Question prompt includes task context")
    else:
        print(f"✗ Question prompt missing task context")
        return False
    
    return True

if __name__ == "__main__":
    results = []
    
    # Run tests
    results.append(("Heuristic Parsing", test_heuristic_parsing()))
    results.append(("Prompt Formats", test_prompt_formats()))
    results.append(("Evaluator Session", test_evaluator_session()))
    
    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    sys.exit(0 if all_passed else 1)
