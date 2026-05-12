import sys
sys.path.append('src')
from evaluator.prompts import extract_score, parse_analysis_response
from evaluator.scoring import MasteryScorer
from evaluator.schemas import LLMAnalysisResponse

# Test improved score extraction
print("=== Testing Score Extraction ===")

test_cases = [
    "Score: 0.85\nStrengths: Good understanding\nWeaknesses: Minor gaps",
    "Score: 0.\nStrengths: Some knowledge",  # Incomplete score
    "Score: 1.0\nMissing Concepts: none",
    "No score here, just feedback",  # No score
    "Score: 0.75",  # Valid score
]

for i, text in enumerate(test_cases):
    score = extract_score(text)
    print(f"Test {i+1}: '{text[:50]}...' -> Score: {score}")

print("\n=== Testing Analysis Parsing ===")

analysis_text = """Score: 0.8
Strengths: Good understanding of basic concepts
Weaknesses: Missing some details
Missing Concepts: neural network, algorithm"""

parsed = parse_analysis_response(analysis_text)
print(f"Parsed analysis: {parsed}")

print("\n=== Testing Mastery Score Computation ===")

# Test with valid LLM score
analysis1 = LLMAnalysisResponse(
    concept_coverage=0.6,
    logical_coherence=0.5,
    causal_reasoning=0.5,
    error_awareness=0.5,
    answer_feedback="Score: 0.8\nStrengths: Good work\nWeaknesses: Minor issues",
    guessing_detected=False,
    missing_concepts=["neural network"],
    misconceptions=[]
)

score1 = MasteryScorer.compute_mastery_score(analysis1, concept_score=0.7)
print(f"Valid LLM score: {score1} ({int(score1 * 100)}%)")

# Test with failed LLM parsing (fallback)
analysis2 = LLMAnalysisResponse(
    concept_coverage=0.6,
    logical_coherence=0.5,
    causal_reasoning=0.5,
    error_awareness=0.5,
    answer_feedback="Good work but missing some concepts",  # No score
    guessing_detected=False,
    missing_concepts=["neural network", "algorithm"],
    misconceptions=[]
)

score2 = MasteryScorer.compute_mastery_score(analysis2, concept_score=0.7, last_valid_score=0.6)
print(f"LLM parsing failed (fallback): {score2} ({int(score2 * 100)}%)")

print("\n=== Testing Missing Concept Feedback ===")

feedback = MasteryScorer.generate_missing_concept_feedback(
    mastery_score=0.65,
    missing_concepts=["neural network", "algorithm", "data processing"],
    key_concepts=["machine learning", "neural network", "algorithm", "data processing"],
    threshold=0.7
)
print(f"Missing concept feedback: {feedback}")

print("\n=== All tests completed ===")