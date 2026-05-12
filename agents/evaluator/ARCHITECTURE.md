"""
ARCHITECTURE DOCUMENT
EvaluatorAgent: Production-Ready Student Evaluation System

This document provides a detailed overview of the system design,
component interactions, and key algorithms.
"""

# =============================================================================
# 1. SYSTEM OVERVIEW
# =============================================================================

"""
The EvaluatorAgent is a comprehensive evaluation system that assesses whether
students truly understand and can apply completed tasks through:

1. Adaptive Socratic questioning at multiple depth levels
2. Deterministic multi-dimensional answer scoring
3. Guessing pattern detection
4. Mastery threshold evaluation
5. Personalized feedback and recommendations

Key Design Principle:
  Clean Architecture with clear separation between domain logic (scoring,
  analysis) and orchestration (state management, workflow).
"""

# =============================================================================
# 2. COMPONENT ARCHITECTURE
# =============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                         EvaluatorAgent (Orchestrator)                    │
│  - Manages evaluation lifecycle                                          │
│  - Coordinates all services                                              │
│  - Maintains evaluation context                                          │
└─────────────────┬───────────────────────────────────────────────────────┘
                  │
        ┌─────────┼──────────┬──────────────┬──────────────┐
        │         │          │              │              │
        ▼         ▼          ▼              ▼              ▼
    ┌────────┐ ┌──────────┐ ┌────────────┐ ┌───────────┐ ┌──────────────┐
    │LLM     │ │Blueprint │ │Socratic    │ │Answer     │ │Mastery       │
    │Interface│ │Builder   │ │Generator   │ │Analyzer   │ │Scorer        │
    └────────┘ └──────────┘ └────────────┘ └───────────┘ └──────────────┘
        │
        ▼
    ┌─────────────────────┐
    │  MockLLMProvider    │ (Can be swapped for any implementation)
    └─────────────────────┘

Supporting Systems:
├── State Machine: Validates state transitions
├── Models: Pydantic schemas for type safety
└── Context: Maintains evaluation state across turns
"""

# =============================================================================
# 3. DATA FLOW ARCHITECTURE
# =============================================================================

"""
INITIALIZATION PHASE:
┌──────────────────────────────────────────────────────────────────┐
│ 1. User calls initialize_evaluation(task_specs)                   │
│ 2. BlueprintBuilder creates comprehensive evaluation blueprint    │
│ 3. Context initialized with empty history                         │
│ 4. SocraticQuestionGenerator creates first question               │
│ 5. Return SocraticQuestion to user                                │
└──────────────────────────────────────────────────────────────────┘
                            ▼
EVALUATION LOOP (repeats until mastery/failure):
┌──────────────────────────────────────────────────────────────────┐
│ 1. User provides student_answer                                   │
│ 2. Answer recorded in context.answer_history                      │
│ 3. AnswerAnalyzer scores across 5 dimensions                      │
│ 4. Analysis recorded in context.analysis_history                  │
│ 5. MasteryScorer computes overall mastery_score                   │
│ 6. Check against thresholds:                                      │
│    - score >= 0.85 → Return SUCCESS with reward                   │
│    - score < 0.60 → Return FAILURE with reschedule               │
│    - 0.60-0.84 → Return CONTINUE with next question              │
└──────────────────────────────────────────────────────────────────┘
"""

# =============================================================================
# 4. SCORING ALGORITHM DETAILS
# =============================================================================

"""
ANSWER ANALYSIS - 5 Dimensions:

1. CONCEPT COVERAGE SCORE (0-1)
   ────────────────────────────
   Measures: How many required concepts are mentioned
   Calculation:
     - Count matches of blueprint.content_dimension.concepts in answer
     - coverage = matches / total_concepts
     - Boost by 1.2x (capped at 1.0) to favor comprehensive answers
   
   Example: If concepts = ["recursion", "base case", "stack"]
           and answer mentions 2 concepts:
           score = min((2/3) * 1.2, 1.0) = 0.8

2. LOGICAL COHERENCE SCORE (0-1)
   ─────────────────────────────
   Measures: Flow and clarity of explanation
   Calculation:
     - Count coherence indicators: "because", "therefore", "first", "then"
     - base_score = min(found_indicators / 3, 1.0)
     - Penalize short answers (< 2 sentences) by 0.7x
     - Enforce minimum 0.2
   
   Example: 3-sentence answer with 2 connectors:
           score = min(2/3, 1.0) = 0.67

3. CAUSAL EXPLANATION SCORE (0-1)
   ──────────────────────────────
   Measures: Presence of "why" and causal reasoning
   Keywords: "because", "since", "reason", "caused", "results in"
   Calculation:
     - Count keyword appearances
     - score = min((found / num_keywords) * 2, 1.0)
     - Enforce minimum 0.1
   
   Example: Answer with 3 causal keywords out of 10:
           score = min((3/10) * 2, 1.0) = 0.6

4. ERROR AWARENESS SCORE (0-1)
   ──────────────────────────
   Measures: Understanding of edge cases and potential errors
   Calculation:
     - Count error phrases: "edge case", "error", "issue", "but", "however"
     - Count mentioned edge cases from blueprint
     - score = base_score + edge_case_bonus
     - base_score = min((found_phrases / 10) * 1.5, 1.0)
     - edge_case_bonus = min(cases_mentioned * 0.15, 0.4)
   
   Example: 3 error phrases + 2 edge cases mentioned:
           score = min((3/10)*1.5, 1.0) + min(2*0.15, 0.4) = 0.45 + 0.3 = 0.75

5. SPECIFICITY SCORE (0-1)
   ───────────────────────
   Measures: Concrete examples vs generic language
   Calculation:
     - Count generic phrases: "maybe", "sort of", "depends"
     - Count specific indicators: "example", "specifically", "for instance"
     - score = 0.5 - (generic_count * 0.15) + (specific_count * 0.1)
     - Clamp to [0.1, 1.0]
   
   Example: 2 generic phrases, 1 specific indicator:
           score = 0.5 - 0.3 + 0.1 = 0.3

GUESSING DETECTION PATTERNS:
────────────────────────────
Flag "excessive_generic_language" if generic_count >= 3
Flag "definition_only_no_application" if definitions > 0 AND length < 50
Flag "lacks_concrete_examples" if "example" not in answer AND length > 20
Flag "suspiciously_brief" if length < 10
Flag "explicit_uncertainty" if contains "not sure", "don't know", etc.

MASTERY SCORE COMPUTATION:
──────────────────────────
overall_mastery = 0.4 × concept_coverage
                + 0.3 × process_clarity
                + 0.2 × causal_explanation
                + 0.1 × error_awareness

Where:
  process_clarity = (specificity_score + coherence_score) / 2

Weighting Rationale:
  - 40% Content: Most important - did student learn concepts?
  - 30% Process: Important - can they articulate their approach?
  - 20% Causal: Important - do they understand cause-effect?
  - 10% Error: Nice-to-have - do they consider edge cases?
"""

# =============================================================================
# 5. STATE MACHINE
# =============================================================================

"""
STATE TRANSITIONS:

DEFINE_BLUEPRINT
    │
    ├──→ ASK_QUESTION
            │
            ├──→ ANALYZE_ANSWER
                    │
                    ├──→ CHECK_MASTERY
                            │
                            ├──→ REWARD (mastery >= 0.85)
                            │       └──→ [TERMINAL]
                            │
                            ├──→ FAIL (mastery < 0.60)
                            │       └──→ [TERMINAL]
                            │
                            └──→ ASK_QUESTION (0.60 <= mastery < 0.85)
                                    └──→ [Loop back]

Valid Transitions:
┌─────────────────────┬──────────────────────────────────┐
│ Current State       │ Valid Next States                │
├─────────────────────┼──────────────────────────────────┤
│ DEFINE_BLUEPRINT    │ ASK_QUESTION                     │
│ ASK_QUESTION        │ ANALYZE_ANSWER                   │
│ ANALYZE_ANSWER      │ CHECK_MASTERY                    │
│ CHECK_MASTERY       │ REWARD, FAIL, ASK_QUESTION      │
│ REWARD              │ [none - terminal]                │
│ FAIL                │ [none - terminal]                │
└─────────────────────┴──────────────────────────────────┘

State Machine Benefits:
- Prevents invalid state transitions
- Makes evaluation flow explicit
- Enables error detection
- Simplifies testing
"""

# =============================================================================
# 6. QUESTION GENERATION STRATEGY
# =============================================================================

"""
DEPTH PROGRESSION:
┌──────────┬──────────┬────────────┬──────────────────┐
│ WHAT     │ WHY      │ WHAT_IF    │ GENERALIZATION   │
├──────────┼──────────┼────────────┼──────────────────┤
│ Describe │ Explain  │ Challenge  │ Apply beyond     │
│ basic    │ reasoning│ assumptions│ context          │
│          │          │            │                  │
│ L1       │ L2       │ L3         │ L4               │
└──────────┴──────────┴────────────┴──────────────────┘

FOCUS AREA SELECTION:
Algorithm: Least-Recently-Covered
  1. Track all focus areas from blueprint
  2. Count previous questions about each area
  3. Select area with minimum count
  4. Round-robin across areas for balanced coverage

DIMENSION SELECTION:
Algorithm: Balanced Alternation
  1. Count previous "process" dimension questions
  2. Count previous "content" dimension questions
  3. Select dimension with fewer questions
  4. Ensures holistic evaluation

DIFFICULTY ESCALATION:
Base Difficulty = {
  WHAT: 1,
  WHY: 2,
  WHAT_IF: 3,
  GENERALIZATION: 4,
}

Adaptive Adjustment:
  - If last 2 answers avg > 0.85: difficulty += 1 (max 5)
  - If last 2 answers avg < 0.50: difficulty -= 1 (min 1)
  - Keeps student in productive struggle zone
"""

# =============================================================================
# 7. FEEDBACK AND RECOMMENDATIONS
# =============================================================================

"""
ON PARTIAL MASTERY (0.60-0.84):
  - Continue with next question
  - Escalate depth if score >= 0.75
  - Maintain focus on weak areas

ON FAILURE (< 0.60):
  Weak Areas Identification:
    - Low concept_coverage → weak_concepts
    - Low causal_explanation → weak_concepts
    - Multiple low scores → failed_process_steps

Recommended Action Selection:
  ┌───────────────────────────────────────────┐
  │ If guessing_count >= 3:                   │
  │   → REVIEW (review materials)             │
  │                                            │
  │ If mastery_score < 0.40:                  │
  │   → BREAK_DOWN (simplify task)            │
  │                                            │
  │ Otherwise:                                 │
  │   → SIMPLIFY (focus on core concepts)     │
  └───────────────────────────────────────────┘

ON SUCCESS (>= 0.85):
  Reward Computation:
    base_points = 100
    bonus = int((mastery_score - 0.85) * 300)
    learning_points = base_points + bonus
    
    streak_multiplier = {
      >= 0.95: 3x,
      >= 0.90: 2x,
      otherwise: 1x
    }
    
    Example: mastery = 0.92
      points = 100 + int((0.92 - 0.85) * 300) = 100 + 21 = 121
      multiplier = 2x
"""

# =============================================================================
# 8. SERVICE INTERFACES
# =============================================================================

"""
LLM INTERFACE (Abstract):
─────────────────────
Purpose: Decouple evaluation from specific LLM implementation
Can be implemented with:
  - OpenAI GPT-4
  - Anthropic Claude
  - Local models (Llama, Mistral)
  - Mock provider for testing

Methods:
  1. generate_socratic_question(blueprint, history, depth, dimension)
     - Takes: task context + history
     - Returns: SocraticQuestion with refined wording
     
  2. assess_answer_quality(question, answer, concepts)
     - Takes: question + response + expected concepts
     - Returns: dict with semantic analysis
     
Benefits:
  - Flexibility in LLM choice
  - Easy testing with MockLLMProvider
  - Can add custom preprocessing
"""

# =============================================================================
# 9. CONTEXT MANAGEMENT
# =============================================================================

"""
Evaluation Context (maintains across multiple turns):

context = EvaluationContext(
    blueprint = EvaluationBlueprint,         # Never changes
    question_history = [SocraticQuestion],   # Accumulates
    answer_history = [str],                  # Accumulates
    analysis_history = [AnswerAnalysis],     # Accumulates
    current_depth_level = DepthLevel,        # May escalate
    guessing_count = int,                    # Increments
)

Why Context is Important:
  - Enables multi-turn evaluation
  - Tracks coverage of all focus areas
  - Prevents repetitive questioning
  - Maintains student trajectory
  - Supports escalation decisions
"""

# =============================================================================
# 10. ERROR HANDLING STRATEGY
# =============================================================================

"""
Validation Points:

1. Input Validation (Pydantic):
   - All models validate on creation
   - Type errors caught immediately
   - Required fields enforced

2. State Validation:
   - StateMachine.validate_transition() prevents invalid moves
   - Raises ValueError on invalid transition
   - Maintains evaluation integrity

3. Context Validation:
   - Check context is initialized before submit_answer()
   - Check blueprint exists before analysis
   - Prevent null references

4. Score Validation:
   - All scores clamped to [0, 1]
   - Formulas prevent overflow
   - Thresholds always valid
"""

# =============================================================================
# 11. TESTABILITY DESIGN
# =============================================================================

"""
Testable Characteristics:

1. Deterministic Logic:
   - No randomness in scoring
   - Same inputs always produce same outputs
   - Easy to verify with fixed test data

2. Dependency Injection:
   - LLM is injected, not created in agent
   - Mock implementation available
   - Easy to replace for testing

3. Pure Functions:
   - Most methods are side-effect free
   - Functions can be tested independently
   - Composition is transparent

4. Data-Driven Tests:
   - Test with various answer quality levels
   - Verify scoring across edge cases
   - Test state transitions exhaustively

Test Coverage Areas:
  ✓ Blueprint construction
  ✓ Question generation
  ✓ Answer analysis and scoring
  ✓ Mastery computation
  ✓ State transitions
  ✓ Complete evaluation flows
"""

# =============================================================================
# 12. PERFORMANCE CONSIDERATIONS
# =============================================================================

"""
Time Complexity:
  - Initialization: O(1)
  - Question generation: O(n) where n = previous questions
  - Answer analysis: O(m) where m = answer length in words
  - Mastery scoring: O(k) where k = answer history length
  - Overall per turn: O(n + m + k) typically << 100ms

Space Complexity:
  - Blueprint storage: O(c + p) where c = concepts, p = steps
  - History storage: O(k) where k = number of turns
  - Question/answer storage: O(m + n) typical per turn

Optimization Opportunities:
  - Cache concept/principle lists for repeated lookups
  - Batch multiple analysis dimensions
  - Stream-process long answers if needed
"""

# =============================================================================
# 13. EXTENSIBILITY POINTS
# =============================================================================

"""
1. Custom LLM Providers:
   class MyLLM(LLMInterface):
       def generate_socratic_question(...): ...
       def assess_answer_quality(...): ...

2. Custom Scoring Weights:
   # In mastery_scorer.py, modify weights
   overall = 0.5*content + 0.2*process + 0.2*causal + 0.1*error

3. Additional Guessing Detection:
   # In answer_analyzer.py, add patterns
   if suspicious_pattern in answer:
       indicators.append("custom_indicator")

4. Custom Feedback Generation:
   # In mastery_scorer.py, enhance feedback
   def _enhanced_feedback(...):
       # Generate personalized feedback

5. Persistent Storage:
   # Wrap agent with storage layer
   class PersistentEvaluator:
       def save_context(self, context):
       def load_context(self, session_id):
"""

# =============================================================================
# 14. DEPLOYMENT CONSIDERATIONS
# =============================================================================

"""
Production Checklist:

[ ] Install with: pip install -r requirements.txt
[ ] Run tests: pytest test_evaluator.py -v
[ ] Configure LLM provider (replace MockLLMProvider)
[ ] Add logging: logging.getLogger(__name__)
[ ] Add metrics/monitoring
[ ] Handle rate limits if using API
[ ] Add request timeout handling
[ ] Validate against student population
[ ] Monitor mastery threshold calibration
[ ] Collect evaluation quality metrics
[ ] A/B test scoring weights
[ ] Version evaluation blueprints
[ ] Archive evaluation history
"""

# =============================================================================
# END OF ARCHITECTURE DOCUMENT
# =============================================================================
