**Agents Data Flow & Structure Report**

- **Scope:** This document summarizes the new/modified agents, services, data models, and runtime data flows introduced during the recent enhancement work. It focuses on architecture, persistence, observability, and integration points between agents and services.

**Executive Summary:**
- Introduced a production-ready vector store pipeline (embedder + FAISS adapter + MongoDB persistence) to support retrieval-augmented flows for ingestion and planning.
- Added persistent coach history and pacing memory to enable personalized, adaptive decisions across sessions.
- Implemented signal smoothing (EMA) + trend computation and wired hourly focus profiles into scheduling heuristics.
- Reworked planner retrieval to use precomputed embeddings, disk-backed FAISS indices, and a three-path load strategy to avoid redundant encoding.
- Replaced ad-hoc prints with structured JSON logging (`utils/logger.py`) and added trace-id propagation across API → orchestrator → agents → LLM calls.

**Components (summary)**

- utils/logger.py
  - JSONFormatter + `get_logger(name)` used across agents/services.

- services/vector_store/embedder.py
  - `EmbeddingService` singleton wrapping SentenceTransformers `all-MiniLM-L6-v2`. Returns L2-normalised float32 vectors.

- services/vector_store/adapter.py
  - `VectorStoreAdapter` manages in-memory FAISS indices, MongoDB `chunk_embeddings` collection, disk persistence (FAISS `.index` files). Exposes `add_course`, `search`, `load_course`, `delete_course`.

- agents/course_ingestion/enrichment/chunk_embedder.py
  - `embed_all_subtopics(subtopics)` — bulk-encodes chunks in a single pass for efficiency and stores `chunk_embeddings` on the subtopic objects.

- agents/course_ingestion/enrichment/deduplicator.py
  - `deduplicate_chunks(chunks, embeddings, threshold=0.95)` — greedy O(N^2) duplicate removal using cosine similarity on L2-normalised vectors.

- agents/planner/rag/index_store.py
  - Disk helpers for saving/loading FAISS indices and chunk lists. Supports save/load/delete by `course_id` with a controlled directory.

- agents/planner/rag/retriever.py
  - Added `add_precomputed_embeddings(chunks, embeddings)` to accept precomputed vectors and bypass re-encoding.

- agents/planner/agent.py
  - `_load_retrieval_context()` implements three-path strategy: 1) precomputed embeddings in course doc → load; 2) disk FAISS index → load; 3) fallback re-encode text and build index. Saves FAISS to disk after plan build.

- agents/planner/memory/pacing_store.py
  - MongoDB-backed rolling-median of actual/estimated durations per-user (and per-subject fallback). Exposes `get_user_pacing_factor(user_id, subject_tag)` and `record_task_completion(...)`.

- agents/coach/services/coach_history_repository.py
  - `CoachHistoryRepository` persists `coach_actions` documents with `user_id`, `trace_id`, `ts`, `action_type`, `message`, `reasoning`, `focus_state`, `fatigue_state`, `affective_state`, and key context fields. Graceful no-op when DB unavailable.

- agents/coach/models/schemas.py
  - `CoachInput` extended with `current_task_title`, `current_task_difficulty`, `current_task_subject`, `current_task_key_concepts` and requires `fatigue_state` (Pydantic model).

- agents/coach/decision/prompt.py
  - `build_user_prompt(context_json, recent_history=None, task_context=None)` — composes prompt with recent actions and current task context, increasing LLM context relevance.

- agents/coach/decision/llm_decider.py
  - `decide_with_llm()` accepts `recent_history` + `trace_id`, calls LLM with structured prompt, logs parse errors and returns `CoachAction`.

- agents/coach/rules/rule_engine.py
  - Added Rule 6: `_rule_declining_trend()` — triggers a `nudge` when `focus_state == "Drifting"` and `signals.focus_trend < -0.4`.

- agents/coach/agent.py
  - `run_coach(input_data, user_id="", trace_id="")` loads `recent_history` from `CoachHistoryRepository`, applies rules, falls back to LLM decider, and persists actions.

- services/signal_processing_service/smoothing.py
  - `EMAState` singleton for per-user EMA of focus/fatigue, thread-safe, exposes `update/get/reset` and `n_updates` counter.

- services/signal_processing_service/signal_snapshot.py
  - `SignalSnapshot` extended with `focus_trend: Optional[float]` to carry short-term trend metrics alongside snapshot values.

- services/signal_processing_service/repository.py
  - `get_hourly_focus_profile(user_id, days_back=30)` — aggregation by hour-of-day averaging `focus_score` to produce a 0-23 profile.  
  - `compute_focus_trend(user_id, window_minutes=5)` — recent-time linear regression (slope) to detect short-term declines.

- agents/scheduler/services/scheduling_heuristics.py
  - `score_slot()` now accepts `hourly_focus_profile` and applies a focus bonus/penalty per slot (±0.15 max) to bias scheduling toward high-focus hours.

- agents/scheduler/agent.py
  - `SchedulingContext` now carries `user_id` and `hourly_focus_profile`. The scheduler loads the profile (if `user_id` present) once and uses it when scoring slots.

- services/ai_orchestrator/orchestrator.py
  - Parallelized I/O (using `ThreadPoolExecutor`) for fetching tasks and recent signals, generates `trace_id` if missing and propagates it to the coach run.

- services/api/main.py
  - FastAPI endpoints updated to accept/forward `x-trace-id` header to orchestrator and added three endpoints: `/planner/record-completion`, `/scheduler/reschedule`, `/signals/calibrate`.

**Data Models & MongoDB Collections**
- `chunk_embeddings` — stores per-course per-chunk metadata and optionally embeddings (for fallback). Used by `VectorStoreAdapter`.
- `coach_actions` — persisted coach events for history/context and offline analytics.
- `pacing_data` — per-user historical actual vs estimated duration ratios used by `PacingStore`.
- `courses` — unchanged semantic course documents now may include `chunk_embeddings` field (list of per-subtopic embeddings).

**End-to-end Retrieval Flow (ingest → plan → retrieve)**
1. Course ingestion tokenizes course into subtopics → `embed_all_subtopics()` encodes chunk vectors once and writes `chunk_embeddings` into the course JSON.
2. `deduplicate_chunks()` removes near-duplicate chunks before persistence.
3. `DatabaseService.save_course()` persists the course JSON including `chunk_embeddings`.
4. Planner `_load_retrieval_context(course_id)` attempts:
   - (A) Load precomputed embeddings from `courses` document → call `retriever.add_precomputed_embeddings()` (fast path).
   - (B) If absent, load disk FAISS index via `index_store.load_index(course_id)` (index path persisted per course) → fast local retrieval.
   - (C) Otherwise re-encode chunks via `EmbeddingService` and build FAISS index, then persist it to disk and optionally to MongoDB for future runs.
5. Retrieval queries use FAISS `IndexFlatIP` with L2-normalised vectors (dot product == cosine similarity) to return nearest chunks and their metadata for RAG.

**Coach Decisioning Flow**
1. API or orchestrator calls `run_coach()` with `CoachInput` and `trace_id`.
2. Agent loads recent `coach_actions` from `CoachHistoryRepository.get_recent_actions()` and computes short-term signals (trend via `SignalRepository.compute_focus_trend()` if available).
3. Rule engine runs quick deterministic checks (including the new declining-trend rule).
4. If no rule deterministically decides, `llm_decider.decide_with_llm()` is called with recent history and task context; the `trace_id` is propagated to the LLM request for traceability.
5. The resulting `CoachAction` is persisted to `coach_actions` and returned.

**Scheduler + Signals Integration**
- `SignalRepository.get_hourly_focus_profile(user_id)` produces a 24-bin profile used to bias `score_slot()`.
- `compute_focus_trend()` returns recent slope; if negative beyond threshold, coach rules may act preemptively.
- EMA smoothing ensures noisy frame-by-frame focus estimates do not cause oscillatory behavior.

**Observability & Debugging**
- Structured JSON logs across all agents ensure machine-parseable logs.
- `x-trace-id` header is accepted at API entry and threaded through orchestrator → agent → LLM for request correlation.
- Key events persisted: `coach_actions`, pacing records, FAISS index saves.

**Testing & CI notes**
- Unit tests: all agent/unit tests pass locally (51 passed, 1 integration skipped). The integration test that was skipped requires a seeded MongoDB.
- Tests added/updated: coach history, pacing store, retriever, chunk embedder, EMA smoothing.

**Operational Considerations**
- FAISS index persistence: indices are stored to disk under a configurable `FAISS_INDEX_DIR`. Ensure the runtime has sufficient disk and that indices are included in backup/restore policies if needed.
- Embeddings: `all-MiniLM-L6-v2` may be swapped for other models; ensure dimensionality/config compatibility when swapping.
- Database availability: Several components gracefully no-op when MongoDB is unavailable; monitor logs for skipped persistence.
- LLM provider: `google.generativeai` usage shows a deprecation warning; migrate to `google-genai` or another supported LLM SDK in future work.

**Files of interest**
- [agents/course_ingestion/enrichment/chunk_embedder.py](agents/course_ingestion/enrichment/chunk_embedder.py)
- [agents/course_ingestion/enrichment/deduplicator.py](agents/course_ingestion/enrichment/deduplicator.py)
- [services/vector_store/embedder.py](services/vector_store/embedder.py)
- [services/vector_store/adapter.py](services/vector_store/adapter.py)
- [agents/planner/rag/index_store.py](agents/planner/rag/index_store.py)
- [agents/planner/rag/retriever.py](agents/planner/rag/retriever.py)
- [agents/coach/services/coach_history_repository.py](agents/coach/services/coach_history_repository.py)
- [agents/planner/memory/pacing_store.py](agents/planner/memory/pacing_store.py)
- [services/signal_processing_service/smoothing.py](services/signal_processing_service/smoothing.py)
- [services/signal_processing_service/repository.py](services/signal_processing_service/repository.py)
- [agents/scheduler/services/scheduling_heuristics.py](agents/scheduler/services/scheduling_heuristics.py)
- [services/ai_orchestrator/orchestrator.py](services/ai_orchestrator/orchestrator.py)
- [services/api/main.py](services/api/main.py)

**Next recommended actions**
- Seed a test MongoDB with representative `courses` and run the skipped integration to validate end-to-end planner→scheduler behavior.
- Add monitoring dashboards for FAISS index sizes and pacing_data growth.
- Migrate deprecated `google.generativeai` usage to `google-genai`.
- Add end-to-end smoke tests covering ingestion → planning → scheduling with a CI-hosted MongoDB instance.

---

Report generated automatically. If you want this exported to a different path or a PDF, tell me where and I'll produce it.