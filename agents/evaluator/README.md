# Evaluator Agent

Socratic evaluation agent that assesses student mastery through adaptive, multi-turn
questioning. Evaluations run step-by-step through the job bus via `EvaluatorWorker`
(see `workers/evaluator_worker.py`); the packaged agent is HTTP-free.

## Architecture

Flat package under `agents/evaluator/`, consistent with the planner/coach agents:

```
agents/evaluator/
├── __init__.py        # light package (lazy heavy imports)
├── agent.py           # EvaluatorAgent — multi-turn Socratic orchestration
├── llm_client.py      # GeminiClient shim over shared LiteLLM `evaluator` group
├── prompts.py         # prompt builders + deterministic helpers
├── schemas.py         # pydantic models (session, analysis, state)
├── scoring.py         # MasteryScorer — deterministic scoring + thresholds
├── state_machine.py   # StateMachine — valid evaluation transitions
└── tests/             # pytest suite (agent, scoring, state machine)
```

### LLM wiring

All LLM calls route through the shared client in `utils/llm_client.py`, which uses
LiteLLM and the `evaluator` model group in `litellm/config.yaml` (same as coach,
planner, search, etc.). There is no direct provider SDK usage.

- When `LLM_MOCK=1` or no provider key is available, `ask("evaluator", ...)` uses a
  supplied `mock_fn`; the evaluator falls back to local concept-coverage scoring.
- `require_llm=False` builds the agent with no LLM for deterministic tests.

### Session state

Sessions and their multi-turn state are held in-memory keyed by `session_id`. There is
no database dependency. Cross-restart rehydration is supplied by the worker's state
store (EVAL-02), not by this package.

## Multi-turn evaluation flow

1. `start_session(task_title, task_description, task_details, max_attempts)` creates a
   session and returns the first Socratic question (WHAT depth).
2. `handle_user_answer(session_id, answer)` analyzes the answer, computes a mastery
   score, and escalates depth WHAT → WHY → HOW while score is below mastery.
3. Terminal when mastery is confirmed, the answer fails, or `max_attempts` is reached.

Deterministic scoring weights and thresholds live in `scoring.py`; the retained
transition rules live in `state_machine.py`.

## HTTP exposure

The evaluator exposes no HTTP endpoints. The legacy `/api/ai/evaluator/*` routes and
the orchestrator's direct `EvaluatorAgent` path were removed (EVAL-01). Evaluation
steps are driven by the `study.eval.step` job consumed by `EvaluatorWorker`.

## Run the tests

```bash
pytest agents/evaluator/tests/ -v
```
