"""Real CoachWorker for the COACH-11 round-trip integration test.

Spawns the actual coach worker (not a stub) so the full nudge loop is proven
through a real broker: Node POST /api/v1/coach/nudge → `study.coach.nudge`
job → CoachWorker consumes → AIOrchestrator decides (LLM mocked via
LLM_MOCK=1 / the coach canned mock responder) → the COACH-09 idempotent
history row is written → result published → Node result consumer marks the
AiJob COMPLETED with the nudge.

Env:
  LLM_MOCK=1        force the canned mock decision responder (no real key)
  RABBITMQ_URL      broker url (passed through from the jest harness)
"""

from __future__ import annotations

import asyncio
import os
import signal


def main() -> None:
    os.environ.setdefault("LLM_MOCK", "1")

    from workers.coach_worker import CoachWorker

    async def run() -> None:
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:  # pragma: no cover (non-unix)
                pass
        worker = CoachWorker()
        await worker.start()
        print(f"STUB_WORKER_READY {CoachWorker.job_type}", flush=True)
        await stop.wait()
        await worker.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()