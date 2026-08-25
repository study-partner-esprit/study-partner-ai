"""Stub AI workers for the AI-COM-10 round-trip integration test.

Runs a BaseAIWorker with a mocked handler — no LLM, no DB. Behavior and job
type are selected via env so the Node jest harness can drive both the happy
path and the retry/DLQ path against a real broker.

Env:
  JOB_TYPE   one of the registered types (default study.plan.generate)
  BEHAVIOR   ok | fail_retryable
"""

from __future__ import annotations

import asyncio
import os
import signal


def main() -> None:
    job_type = os.getenv("JOB_TYPE", "study.plan.generate")
    behavior = os.getenv("BEHAVIOR", "ok")

    from messaging.failures import RetryableError
    from workers.base import BaseAIWorker

    class StubWorker(BaseAIWorker):
        async def handle(self, payload, envelope):
            if behavior == "fail_retryable":
                raise RetryableError("LLM timeout (stub)")
            return {
                "plan": {
                    "tasks": [
                        {"title": f"Study: {payload.get('goal', 'goal')}", "duration_minutes": 45}
                    ]
                },
                "stub": True,
            }

    # bind job_type dynamically
    StubWorker.job_type = job_type

    async def run() -> None:
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:  # pragma: no cover (non-unix)
                pass
        worker = StubWorker()
        await worker.start()
        print(f"STUB_WORKER_READY {job_type} {behavior}", flush=True)
        await stop.wait()
        await worker.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
