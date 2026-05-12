# EvaluatorAgent: Production-Ready Student Evaluation System

A comprehensive Python implementation of an intelligent evaluation agent that assesses whether students truly understand scheduled tasks using deterministic scoring, adaptive Socratic questioning, and guessing detection.

## Architecture

### High-Level Flow
```
Audio Input → Whisper (speech-to-text) → EvaluatorAgent → MongoDB (fetch task + course + concepts) → GPT4All (Socratic question/answer analysis) → Deterministic scoring → Reward/Reschedule
```

### Project Structure
```
src/
├── config/
│   └── settings.py              # Mongo URI, model path, constants
├── database/
│   ├── mongo_client.py          # MongoDB connection
│   ├── repositories/
│   │   ├── task_repository.py
│   │   ├── course_repository.py
│   │   └── scheduling_repository.py
├── evaluator/
│   ├── evaluator_agent.py
│   ├── context_builder.py       # Builds evaluation context from MongoDB
│   ├── speech_to_text.py
│   ├── llm_client.py            # GPT4All offline
│   ├── prompts.py
│   ├── scoring.py
│   ├── state_machine.py
│   ├── reward_engine.py
│   └── schemas.py
├── rescheduler/
│   └── rescheduler_agent.py
└── main.py
```

## Features

### 1. **Offline-First Design**
- **GPT4All 2.8.2**: Local LLM inference, no internet required
- **Whisper**: Local speech-to-text, no API calls
- **MongoDB**: Local database for course data
- Zero external dependencies for production use

### 2. **MongoDB Integration**
- Database: `study_partner`
- Collection: `course.study_partner`
- Dynamic schema handling
- Repository pattern with dependency injection

### 3. **Context Builder**
- Fetches task and course data from MongoDB
- Extracts key concepts, definitions, examples
- Builds structured evaluation context
- No hardcoded concepts

### 4. **Deterministic Mastery Scoring**
```
mastery_score = 0.4 × concept_coverage
              + 0.3 × logical_coherence
              + 0.2 × causal_reasoning
              + 0.1 × error_awareness

States:
  ≥ 0.85 & no guessing → MASTERY_CONFIRMED
  < 0.60 → FAILED
  else → CONTINUE
```

### 5. **Multi-Turn Socratic Evaluation**
- Adaptive depth levels: WHAT → WHY → HOW
- LLM generates questions, Python scores answers
- Business logic separated from LLM layer

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set environment variables:
```bash
export MONGO_URI="mongodb://localhost:27017"
export GPT4ALL_MODEL_PATH="/path/to/model"  # optional
```

## Usage

```python
from src.database.mongo_client import MongoDBClient
from src.database.repositories.course_repository import CourseRepository
from src.evaluator.evaluator_agent import EvaluatorAgent

# Initialize
mongo_client = MongoDBClient()
course_repo = CourseRepository(mongo_client)
evaluator = EvaluatorAgent(course_repo=course_repo)

# Evaluate
result = evaluator.evaluate("task_123", "My answer here", "text")
print(result.state, result.mastery_score)
```

## Dependencies

- pymongo>=4.0
- gpt4all==2.8.2
- openai-whisper>=20231117
- pydantic>=2.0
  < 0.60 → FAILED
```

## Architecture

### Modular Separation of Concerns

```
models.py                 # Pydantic schemas for all types
llm_interface.py         # Abstract LLM integration
blueprint_builder.py     # Blueprint construction
answer_analyzer.py       # Answer quality analysis
socratic_generator.py    # Question generation
mastery_scorer.py        # Scoring and thresholds
state_machine.py         # State management
evaluator_agent.py       # Main orchestration
mock_llm.py             # Mock implementation for testing
```

## Usage

### Basic Example

```python
from evaluator_agent import EvaluatorAgent
from mock_llm import MockLLMProvider

# Initialize agent
llm = MockLLMProvider()
evaluator = EvaluatorAgent(llm)

# Start evaluation
first_question = evaluator.initialize_evaluation(
    task_id="task_001",
    task_description="Implement binary search",
    process_steps=["understand", "design", "implement", "test"],
    required_tools=["IDE", "debugger"],
    concepts_to_cover=["binary search", "divide and conquer"],
    principles=["logarithmic complexity", "recursion"],
    edge_cases=["empty array", "not found"],
    applications=["database", "API pagination"],
)

# Get student answer
print(first_question.question_text)
student_answer = input("Student: ")

# Submit and analyze
result = evaluator.submit_answer(student_answer)

# Check result
if result.evaluation_state == "MASTERY_CONFIRMED":
    print(f"✓ Mastery confirmed!")
    print(f"  Points: {result.reward.learning_points}")
    print(f"  Concepts: {result.reward.concept_strengthened}")

elif result.evaluation_state == "FAILED":
    print(f"✗ Evaluation failed")
    print(f"  Weak areas: {result.reschedule_payload.weak_concepts}")
    print(f"  Action: {result.reschedule_payload.recommended_action.value}")

else:  # Continue
    print(f"Next question: {result.question.question_text}")
```

## Return Values

### On Mastery
```json
{
  "evaluation_state": "MASTERY_CONFIRMED",
  "reward": {
    "learning_points": 115,
    "streak_increment": 2,
    "concept_strengthened": ["binary search", "divide and conquer"]
  },
  "mastery_score": 0.92
}
```

### On Failure
```json
{
  "evaluation_state": "FAILED",
  "reschedule_payload": {
    "weak_concepts": ["recursion", "complexity analysis"],
    "failed_process_steps": ["design", "implement"],
    "recommended_action": "BREAK_DOWN"
  },
  "mastery_score": 0.48
}
```

### On Partial Mastery (Continue)
```json
{
  "evaluation_state": "ASK_QUESTION",
  "question": {
    "question_text": "Why is binary search efficient?",
    "depth_level": "why",
    "dimension": "content",
    "focus_area": "efficiency",
    "difficulty_level": 2
  },
  "mastery_score": 0.72
}
```

## Key Design Patterns

### 1. **Clean Architecture**
- Domain logic separated from orchestration
- Clear responsibility boundaries
- Easy to test and extend

### 2. **Deterministic Scoring**
- No randomness in evaluation
- Reproducible results
- Formula-based thresholds

### 3. **State Machine**
- Prevents invalid state transitions
- Clear evaluation flow
- Validates progression

### 4. **Adapter Pattern for LLM**
```python
from llm_interface import LLMInterface

class GPT4Provider(LLMInterface):
    def generate_socratic_question(self, blueprint, history, depth, dimension):
        # Call GPT-4 API
        pass
    
    def assess_answer_quality(self, question, answer, concepts):
        # Use GPT-4 for semantic analysis
        pass
```

### 5. **Dependency Injection**
```python
# Easy to swap implementations
llm = OpenAIProvider()  # or MockLLMProvider()
evaluator = EvaluatorAgent(llm)
```

## Running Tests

```bash
# Install pytest
pip install pytest pydantic

# Run all tests
pytest test_evaluator.py -v

# Run specific test class
pytest test_evaluator.py::TestBlueprintBuilder -v

# Run with coverage
pytest test_evaluator.py --cov=. --cov-report=html
```

## Running Example

```bash
python example_usage.py
```

## Type Hints & Documentation

All code includes:
- Full type hints for parameters and returns
- Comprehensive docstrings
- Clear variable names
- No placeholder comments

## Extending the Agent

### 1. Custom LLM Provider
```python
from llm_interface import LLMInterface

class CustomLLM(LLMInterface):
    def __init__(self, api_key):
        self.api_key = api_key
    
    def generate_socratic_question(self, blueprint, ...):
        # Your implementation
        pass
```

### 2. Custom Scoring
```python
# Modify weights in mastery_scorer.py
mastery = (
    0.5 * content +  # Increased from 0.4
    0.3 * process +
    0.1 * causal +
    0.1 * error
)
```

### 3. Additional Guessing Detection
```python
# Add patterns to answer_analyzer.py
def _detect_guessing(answer):
    indicators = []
    # Add custom detection logic
    return indicators
```

## Performance Characteristics

- **Initialization**: O(1)
- **Question Generation**: O(n) where n = previous questions
- **Answer Analysis**: O(m) where m = answer length
- **Mastery Computation**: O(k) where k = answer history length
- **Memory**: O(k) for maintaining context

## Dependencies

- **pydantic** >= 2.0: Data validation and schema management
- **typing**: Type hints (standard library)
- **enum**: Enumeration support (standard library)
- **pytest** (dev): Testing framework

## Design Principles Applied

✓ SOLID principles (Single Responsibility, Open/Closed, Liskov, Interface Segregation, Dependency Inversion)
✓ Clean Code (meaningful names, small functions, no side effects)
✓ Design Patterns (State, Adapter, Factory, Builder)
✓ Separation of Concerns (domain ≠ orchestration)
✓ Type Safety (full type hints)
✓ Testability (deterministic, injectable dependencies)
✓ Maintainability (clear structure, comprehensive docs)

## Future Enhancements

- Persistent context storage
- Multi-question parallel evaluation
- Custom domain knowledge integration
- Real-time difficulty adjustment
- Student performance analytics
- Integration with learning management systems

---

**Status**: Production-ready  
**Python Version**: 3.10+  
**License**: Open source
