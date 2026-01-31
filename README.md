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
├─ main.py                    # FastAPI entry point
├─ config/                    # Global settings & constants
├─ models/                    # Shared domain models
├─ orchestrator/              # Meta-agent orchestration
├─ agents/                    # Modular agent implementations
│   ├─ planner/
│   ├─ coach/
│   └─ evaluator/
├─ services/                  # Shared services (DB, logging, signals)
└─ utils/                     # Helper functions
```

## Setup

```bash
# Install dependencies
poetry install

# Run the service
poetry run python main.py
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
