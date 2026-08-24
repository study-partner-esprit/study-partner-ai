"""Topology parity tests (TEST-04): Python must match the shared fixture that
the Node side also validates against. Drift fails CI."""

import json
import pathlib

from messaging import topology as t

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
    assert t.delay_queue_name(1000) == naming["sampleDelayQueue1000"]
    assert t.delay_queue_name(16000) == naming["sampleDelayQueue16000"]


def test_retry_policy_matches_fixture():
    fx = load_fixture()
    assert t.RETRY_DELAYS_MS == fx["retryDelaysMs"]
    assert t.MAX_RETRIES == fx["maxRetries"]
    assert t.RETRY_HEADER == fx["headers"]["retryCount"]
    assert t.work_queue_arguments() == fx["queueArguments"]["workQueue"]
    assert t.delay_queue_arguments(1000) == fx["queueArguments"]["delayQueue1000"]
