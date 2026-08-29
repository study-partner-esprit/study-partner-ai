"""COACH-07: shared LiteLLM client (utils/llm_client.py).

The router is driven entirely by `litellm/config.yaml` (model_list of per-agent
deployments + router_settings fallback chains), mirroring the hackership-ai
layout. These tests assert:
- YAML config resolves for every agent group
- env overrides (LLM_CONFIG path, LLM_MOCK) and dummy keys steer mock vs real
- the real path builds guarded system content and per-call knobs
- retries live in the Router; hard failures surface as LLMRequestError
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from utils import llm_client
from utils.llm_client import (
    AGENT_GROUPS,
    LLMRequestError,
    MissingMockResponderError,
    agent_config,
    ask,
    build_router,
    reload_config,
    reset_router,
)
from security.prompt_guard import build_system_block


@pytest.fixture(autouse=True)
def _clean_state():
    reload_config()
    reset_router()
    yield
    reload_config()
    reset_router()


def _with_keys(monkeypatch, **keys):
    monkeypatch.delenv("LLM_MOCK", raising=False)
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in keys.items():
        monkeypatch.setenv(key, value)


# ------------------------------------------------------------ config load #

def test_config_has_every_agent_deployment():
    cfg = llm_client.load_config()
    names = {entry["model_name"] for entry in cfg["model_list"]}
    assert set(AGENT_GROUPS) <= names
    assert {"groq-main", "nvidia-main", "openrouter-main"} <= names


def test_agent_config_resolves_model_and_defaults():
    coach = agent_config("coach")
    assert coach.model == "gemini/gemini-2.0-flash"
    assert coach.temperature == 0.5
    assert coach.max_tokens == 512
    assert "groq/llama-3.3-70b-versatile" in coach.fallback_models


def test_agent_config_fallback_chain_from_yaml():
    planner = agent_config("planner")
    assert planner.model.startswith("nvidia_nim/")
    assert len(planner.fallback_models) >= 1
    evaluator = agent_config("evaluator")
    assert len(evaluator.fallback_models) >= 2


def test_unknown_agent_rejected():
    with pytest.raises(KeyError):
        agent_config("not-an-agent")
    with pytest.raises(KeyError):
        ask("not-an-agent", "sys", "user")


def test_config_path_env_override(tmp_path, monkeypatch):
    alt = tmp_path / "litellm" 
    alt.mkdir()
    (alt / "config.yaml").write_text(
        "model_list:\n- model_name: coach\n  litellm_params:\n    model: groq/llama-3.3-70b-versatile\n"
    )
    monkeypatch.setenv("LLM_CONFIG", str(alt / "config.yaml"))
    reload_config()
    cfg = llm_client.load_config()
    found = [e for e in cfg["model_list"] if e["model_name"] == "coach"]
    assert found[0]["litellm_params"]["model"] == "groq/llama-3.3-70b-versatile"
    reload_config()  # restore for other tests' expectations


# ---------------------------------------------------------------- mocking #

def test_mock_used_when_no_keys(monkeypatch):
    _with_keys(monkeypatch)
    out = ask("coach", "sys", "hello there", mock_fn=lambda up: "MOCK:" + up)
    assert out == "MOCK:hello there"


def test_mock_forced_by_env(monkeypatch):
    _with_keys(monkeypatch, GEMINI_API_KEY="real-key")
    monkeypatch.setenv("LLM_MOCK", "1")
    assert ask("coach", "sys", "hi", mock_fn=lambda up: "m") == "m"


def test_dummy_gemini_key_means_mock(monkeypatch):
    _with_keys(monkeypatch, GEMINI_API_KEY="dummy_key_for_testing")
    # Dummy counts as absent, no real key anywhere → mock.
    assert llm_client._should_mock("coach")
    # A real fallback key is enough to attempt a real call (router will
    # fail over to groq if the primary provider key is missing).
    _with_keys(monkeypatch, GEMINI_API_KEY="dummy_key_for_testing", GROQ_API_KEY="real-key")
    assert not llm_client._should_mock("coach")


def test_missing_mock_fn_raises(monkeypatch):
    _with_keys(monkeypatch)
    with pytest.raises(MissingMockResponderError) as exc:
        ask("coach", "sys", "hi")
    assert "no API key" in str(exc.value)


# -------------------------------------------------------------- real path #

class _FakeCompletion:
    def completion(self, **kwargs):
        self.kwargs = dict(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="assistant-text"))]
        )


def test_real_path_guards_system_and_passes_knobs(monkeypatch):
    _with_keys(monkeypatch, GEMINI_API_KEY="real-key")
    fake = _FakeCompletion()
    with patch.object(llm_client, "build_router", return_value=fake):
        out = ask("coach", "sys prompt", "user prompt", trace_id="tr")
    assert out == "assistant-text"
    messages = fake.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == build_system_block("sys prompt")
    assert messages[1] == {"role": "user", "content": "user prompt"}
    # YAML defaults apply unless explicitly overridden
    assert fake.kwargs["model"] == "coach"
    assert "temperature" not in fake.kwargs
    assert "max_tokens" not in fake.kwargs


def test_real_path_per_call_override(monkeypatch):
    _with_keys(monkeypatch, GEMINI_API_KEY="real-key")
    fake = _FakeCompletion()
    with patch.object(llm_client, "build_router", return_value=fake):
        ask("coach", "sys", "user", temperature=0.9, max_tokens=64)
    assert fake.kwargs["temperature"] == 0.9
    assert fake.kwargs["max_tokens"] == 64


def test_real_path_error_becomes_llm_request_error(monkeypatch):
    _with_keys(monkeypatch, GEMINI_API_KEY="real-key")

    class _Boom:
        def completion(self, **kwargs):
            raise RuntimeError("provider down")

    with patch.object(llm_client, "build_router", return_value=_Boom()):
        with pytest.raises(LLMRequestError) as exc:
            ask("coach", "sys", "user")
    assert "coach" in str(exc.value)


# ----------------------------------------------------------- router wiring #

def test_router_builds_from_yaml():
    router = build_router()
    models = {d["model_name"] for d in router.model_list}
    assert set(AGENT_GROUPS) <= models
    assert router.fallbacks is None or isinstance(router.fallbacks, list)
    reset_router()


def test_reset_router_rebuilds():
    r1 = build_router()
    reset_router()
    r2 = build_router()
    assert r1 is not r2
    reset_router()


# ------------------------------------------------------------- end-to-end #
# ask() must return a coach-style mock payload through the client when keys are
# absent — the behaviour the decider's call_gemini relies on.

def test_coach_decider_produces_mock_json_without_keys(monkeypatch):
    import json as _json

    from agents.coach.decision import llm_decider
    from agents.coach.decision.prompt import SYSTEM_PROMPT

    _with_keys(monkeypatch)
    raw = llm_decider.call_gemini(SYSTEM_PROMPT, "Student state looks focused")
    assert isinstance(raw, str)
    data = _json.loads(raw)
    assert data.get("action_type") in ("nudge", "encourage", "silence", "break", "rest")