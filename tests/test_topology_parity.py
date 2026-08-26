"""Topology parity tests (TEST-04): Python must match the shared fixture that
the Node side also validates against. Drift fails CI."""

import json
import os
import pathlib

# Parity is defined against the CANONICAL ladder — drop any outer-shell
# override before topology evaluates its constants.
os.environ.pop("AI_RETRY_DELAYS_MS", None)

from messaging import topology as t  # noqa: E402

FIXTURE = (
    pathlib.Path(__file__)
    .resolve()
    .parents[2]
    / "docs"
    / "contracts"
    / "topology-fixture.json"
)


def load_fixture():
    return json.loads(FIXTURE.read_text())


def test_exchanges_match_fixture():
    fx = load_fixture()["exchanges"]
    assert t.EXCHANGE_JOBS == fx["jobs"]
    assert t.EXCHANGE_DELAY == fx["delay"]
    assert t.EXCHANGE_DLX == fx["dlx"]
    assert t.EXCHANGE_RESULTS == fx["results"]


def test_queues_and_naming_match_fixture():
    fx = load_fixture()
    assert t.RESULT_QUEUE == fx["queues"]["results"]
    naming = fx["naming"]
    assert t.work_queue_name("study.plan.generate") == naming["sampleWorkQueue"]
    assert t.dlq_queue_name("study.plan.generate") == naming["sampleDlq"]
    assert t.delay_queue_name("study.plan.generate", 1000) == naming["sampleDelayQueue1000"]
    assert t.delay_queue_name("study.plan.generate", 16000) == naming["sampleDelayQueue16000"]
    assert t.retry_routing_key("study.plan.generate", 1000) == naming["sampleRetryKey1000"]


def test_retry_policy_matches_fixture():
    fx = load_fixture()
    assert t.RETRY_DELAYS_MS == fx["retryDelaysMs"]
    assert t.MAX_RETRIES == fx["maxRetries"]
    assert t.RETRY_HEADER == fx["headers"]["retryCount"]
    assert t.work_queue_arguments("study.plan.generate") == fx["queueArguments"]["workQueue"]
    assert t.delay_queue_arguments(1000) == fx["queueArguments"]["delayQueue1000"]
