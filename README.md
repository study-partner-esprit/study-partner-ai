# Study Partner AI

AI-powered study assistance system using multi-agent architecture.

## Architecture

This service implements a multi-agent system with:
- **Planner Agent**: Creates personalized study plans based on goals and constraints
- **Coach Agent**: Provides real-time guidance, motivation, and nudges
- **Evaluator Agent**: Assesses progress and provides constructive feedback
- **Meta Agent (Orchestrator)**: Routes signals and coordinates agent interactions

## Project Structure

```
study-partner-ai/
├─ services/
│   └─ api/
│       └─ main.py            # FastAPI entry point
├─ agents/                    # Modular agent implementations
│   ├─ planner/               # Study plan decomposition
│   ├─ coach/                 # Real-time coaching decisions
│   ├─ scheduler/             # Task scheduling with constraints
│   └─ course_ingestion/      # Course material processing
├─ services/                  # Shared services
│   ├─ ai_orchestrator/       # Agent coordination
│   ├─ signal_processing_service/ # ML signal collection
│   └─ schedule_orchestrator/ # Schedule management
├─ models/                    # Shared domain models
└─ ML/                        # Machine learning models
```

## Setup

```bash
# Install dependencies
poetry install

# Run the AI service
poetry run python services/api/main.py
```

## API

FastAPI service exposing endpoints for agent interactions:
- `POST /api/v1/session` - Process session requests with signals

## Testing

```bash
# Run all tests
poetry run pytest

# Run specific agent tests
poetry run pytest agents/planner/tests/
poetry run pytest agents/coach/tests/
poetry run pytest agents/evaluator/tests/
```

## Development

Each agent is a modular component with its own models, services, and tests.
