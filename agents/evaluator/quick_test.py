#!/usr/bin/env python3
"""
Quick test of the refactored system with a real small model.
Tests the full flow: start -> answer -> fallback handling.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

from src.evaluator.evaluator_agent import EvaluatorAgent

print("\n" + "="*70)
print("QUICK TEST: Small Model Answer Evaluation")
print("="*70)

try:
    agent = EvaluatorAgent()
    
    print("\n1. Starting evaluation session...")
    session_result = agent.start_session(
        task_title="Water Cycle",
        task_description="Understanding how water moves in the environment",
        task_details="The water cycle involves evaporation, condensation, and precipitation. Water evaporates from oceans and lakes, forms clouds, and falls as rain.",
        max_attempts=5
    )
    
    session_id = session_result['session_id']
    print(f"   ✓ Session started: {session_id}")
    print(f"   Question: {session_result['question'][:80]}...")
    
    print("\n2. Submitting an answer...")
    user_answer = "Water evaporates from oceans when heated by the sun, rises into the atmosphere, and condenses into clouds."
    
    result = agent.handle_user_answer(session_id, user_answer)
    
    print(f"   ✓ Answer processed")
    print(f"   State: {result.get('state', 'UNKNOWN')}")
    print(f"   Score: {result.get('mastery_score', 0.0):.1%}")
    print(f"   Feedback: {result.get('feedback', 'N/A')[:80]}...")
    
    # Verify answer was recorded
    session = agent.get_session(session_id)
    if user_answer in session.answer_history:
        print(f"   ✓ Answer recorded in session history")
    else:
        print(f"   ✗ ERROR: Answer NOT in history!")
    
    # Verify session continued (not marked failed on first answer)
    if result.get('state') in ['CONTINUE', 'MASTERY_CONFIRMED']:
        print(f"   ✓ Session continues (not prematurely FAILED)")
    elif result.get('state') == 'FAILED':
        print(f"   ✗ ERROR: Session marked FAILED on first answer!")
    
    print("\n" + "="*70)
    if result.get('state') in ['CONTINUE', 'MASTERY_CONFIRMED']:
        print("✓ TEST PASSED - System working correctly!")
    else:
        print(f"✗ TEST FAILED - Session ended with state: {result.get('state')}")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
