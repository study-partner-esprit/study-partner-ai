# EvaluatorAgent — Architecture

Socratic evaluation is split between deterministic domain logic (scoring, state
transitions) and orchestration (multi-turn session management). The package is a
flat set of modules under `agents/evaluator/` (planner/coach layout), with no HTTP
surface — evaluation steps run through `EvaluatorWorker` on the job bus.

## Components

```
agents/evaluator/
├── agent.py           EvaluatorAgent — orchestrates a multi-turn session
├── llm_client.py      GeminiClient shim -> shared LiteLLM `evaluator` group
├── prompts.py         prompt builders + deterministic template fallbacks
├── schemas.py         pydantic models (EvaluationSession, State, Analysis)
├── scoring.py         MasteryScorer: deterministic scoring + thresholds
└── state_machine.py   StateMachine: valid evaluation transitions
```

## Data flow

```
start_session(task_title, task_description, task_details)
    └─> EvaluationSession created; first Socratic question generated (WHAT)
            │
            ▼
handle_user_answer(session_id, answer)
    └─> answer stored in session history
        └─> analysis computed (concept coverage + LLM semantics when available)
            └─> MasteryScorer.overall_score(analysis)
                ├─ score >= threshold       -> COMPLETE (mastery confirmed)
                ├─ failing threshold        -> FAILED  (terminal)
                └─ else while attempts remain-> next question (deepen WHAT->WHY->HOW)
```

## Scoring

`scoring.py` holds the deterministic multi-dimensional scoring used to compute an
overall mastery score, plus the `concept_coverage` helper that measures how many of
the task's key concepts a student answer mentions. Thresholds for confirmed /
continuing / failed live beside it. Missing-concept feedback is derived directly from
coverage gaps.

## State machine

`state_machine.py` enforces valid transitions for a session and exposes terminal
state detection, keeping the agent's flow valid and testable.

## LLM wiring

`llm_client.py` calls the shared `ask("evaluator", ...)` in `utils/llm_client.py`,
which resolves the `evaluator` model group from `litellm/config.yaml` through
LiteLLM. There is no direct provider SDK usage and no local model embedded in this
package.

- `LLM_MOCK=1` (or missing keys) forces the mock path; a caller `mock_fn` supplies
  canned responses.
- `EvaluatorAgent(require_llm=False)` builds the agent without any LLM, so tests
  exercise the deterministic scoring/state logic with `RequireLLM` disabled. When
  the LLM is unavailable, question generation falls back to local templates and
  analysis to local concept-coverage scoring.

## Session state

`EvaluatorAgent` keeps sessions in memory, keyed by `session_id`. Multi-turn state
(question/answer/analysis history, current depth, attempt count) lives on each
`EvaluationSession`. Rehydration across restarts is provided by the worker's state
store (EVAL-02), not by this package. There is no database dependency.

## Job-bus integration

`workers/evaluator_worker.py` (`EvaluatorWorker`, job type `study.eval.step`)
consumes evaluation step messages, routes them to start/resume, runs the step via
`asyncio.to_thread`, and lets agent/LLM exceptions be classified retryable by the
base worker policy. No HTTP endpoints expose the evaluator.

## Testing

- `tests/test_agent.py` — session lifecycle (start/answer/history/limit/not-found)
- `tests/test_state_machine.py` — transition validity, terminal sinks, thresholds
- `tests/test_scoring.py` — score parsing, MasteryScorer, concept coverage
- `tests/test_evaluator_worker.py` (repo `tests/`) — worker routing, terminal vs
  retryable handling, off-loop dispatch
