"""Shared LLM client (COACH-07).

Every agent talks to the LLM through :func:`ask`. Model wiring lives in
``litellm/config.yaml`` (same shape as the reference hackership-ai repo): a
``model_list`` of deployments keyed by agent alias plus ``router_settings``
(retries + fallback chains). The alias is passed to :func:`ask` — no code change
needed to re-route, add providers, or change fallbacks, just edit the YAML.

Mock behaviour mirrors the old per-agent mocks: when ``LLM_MOCK=1`` or none of
the required provider API keys are present (or a key is the dummy test value),
a caller-supplied ``mock_fn`` is used instead of a real network call. A failing
real call raises :class:`LLMRequestError` so callers can decide fallback.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from security.prompt_guard import build_system_block
from utils.logger import get_logger

logger = get_logger(__name__)

# Provider prefix -> env var holding its API key. LiteLLM reads these itself for
# the real call; we only inspect them to decide whether to use the mock.
PROVIDER_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nvidia_nim": "NVIDIA_API_KEY",
}

DUMMY_KEYS = {"dummy_key_for_testing"}

# Aliases agents may target via ask(). Anything else the YAML declares (e.g.
# fallback providers) is a router-internal deployment.
AGENT_GROUPS = (
    "coach",
    "planner",
    "search",
    "reflection",
    "course_ingestion",
    "evaluator",
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "litellm" / "config.yaml"

MockResponder = Callable[[str], str]


class LLMRequestError(RuntimeError):
    """A real LLM call failed after the router retries were exhausted."""


class MissingMockResponderError(RuntimeError):
    """A mock was requested (no keys available) but no ``mock_fn`` was given."""


@dataclass(frozen=True)
class AgentLLM:
    """Resolved wiring for one agent alias (from litellm/config.yaml)."""

    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    fallback_models: tuple[str, ...] = ()


_config: Optional[Dict[str, Any]] = None
_router: Optional["litellm.Router"] = None


def config_path() -> Path:
    overridden = os.getenv("LLM_CONFIG")
    return Path(overridden) if overridden else DEFAULT_CONFIG_PATH


def load_config() -> Dict[str, Any]:
    """Parse litellm/config.yaml (cached; use reload_config to clear)."""
    global _config
    if _config is None:
        import yaml

        with open(config_path(), "r", encoding="utf-8") as fh:
            _config = yaml.safe_load(fh) or {}
    return _config


def reload_config() -> None:
    """Drop the cached YAML config (used by tests between env changes)."""
    global _config, _router
    _config = None
    _router = None


def _deployment(name: str) -> Dict[str, Any]:
    for entry in load_config().get("model_list", []):
        if entry.get("model_name") == name:
            params = entry.get("litellm_params", {}) or {}
            return {"model": params.get("model", ""), "litellm_params": params}
    raise KeyError(f"no deployment named {name!r} in {config_path()}")


def _fallback_chain(agent: str) -> List[str]:
    cfg = load_config()
    for mapping in cfg.get("router_settings", {}).get("fallbacks", []):
        if agent in mapping:
            return list(mapping[agent])
    return []


def agent_config(agent: str) -> AgentLLM:
    """Resolve an agent's wiring from the YAML (model + defaults + fallbacks)."""
    if agent not in AGENT_GROUPS:
        raise KeyError(f"unknown llm agent: {agent!r} (known: {', '.join(AGENT_GROUPS)})")
    params = _deployment(agent)["litellm_params"]
    return AgentLLM(
        model=params.get("model", ""),
        temperature=params.get("temperature"),
        max_tokens=params.get("max_tokens"),
        fallback_models=tuple(
            _deployment(fb)["litellm_params"].get("model", "") for fb in _fallback_chain(agent)
        ),
    )


def _provider_of(model: str) -> str:
    return model.split("/", 1)[0]


def _required_key_envs(agent: str) -> List[str]:
    """Env vars involved for an agent's primary model and its fallback chain."""
    cfg = agent_config(agent)
    envs: List[str] = []
    for model in (cfg.model, *cfg.fallback_models):
        if not model:
            continue
        key_env = PROVIDER_KEY_ENV.get(_provider_of(model))
        if key_env and key_env not in envs:
            envs.append(key_env)
    return envs


def _keys_available(agent: str) -> bool:
    """Any involved provider key present? (An explicit fallback works if primary
    is key-less, so we only require at least one real key.)"""
    for key_env in _required_key_envs(agent):
        value = os.getenv(key_env, "")
        if value and value not in DUMMY_KEYS:
            return True
    return False


def _should_mock(agent: str) -> bool:
    if os.getenv("LLM_MOCK") == "1":
        return True
    return not _keys_available(agent)


def _extract_text(response) -> str:
    if hasattr(response, "choices"):
        return response.choices[0].message.content
    return response["choices"][0]["message"]["content"]


def build_router():
    """Build (once) the LiteLLM Router straight from litellm/config.yaml."""
    global _router
    if _router is not None:
        return _router

    import litellm

    litellm.drop_params = True
    cfg = load_config()
    router_settings = dict(cfg.get("router_settings", {}) or {})
    fallbacks = router_settings.pop("fallbacks", [])
    retry_kwargs: Dict[str, Any] = {
        key: router_settings.pop(key)
        for key in ("num_retries", "retry_after", "allowed_fails", "cooldown_time")
        if key in router_settings
    }
    for unused in router_settings:
        logger.warning("llm_router_unknown_setting", extra={"key": unused})
    _router = litellm.Router(
        model_list=cfg.get("model_list", []),
        fallbacks=fallbacks,
        **retry_kwargs,
    )
    return _router


def reset_router() -> None:
    """Drop the cached router (used by tests between env/config changes)."""
    global _router
    _router = None


def ask(
    agent: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    mock_fn: Optional[MockResponder] = None,
    trace_id: str = "",
) -> str:
    """Complete a chat for ``agent`` and return the assistant text.

    System content is placed in its own trusted, guarded block — user content is
    never concatenated into it (COACH-03 protection kept for every agent).
    Per-call ``temperature``/``max_tokens`` override the YAML defaults when given
    (normally the deployment's litellm_params carry them).
    """
    if agent not in AGENT_GROUPS:
        raise KeyError(f"unknown llm agent: {agent!r} (known: {', '.join(AGENT_GROUPS)})")
    if _should_mock(agent):
        if mock_fn is None:
            raise MissingMockResponderError(
                f"no API key available for agent {agent!r} and no mock_fn given"
            )
        logger.info(
            "llm_client_mock_response",
            extra={"agent": agent, "trace_id": trace_id},
        )
        return mock_fn(user_prompt)

    messages = [
        {"role": "system", "content": build_system_block(system_prompt)},
        {"role": "user", "content": user_prompt},
    ]
    kwargs: Dict[str, Any] = {"model": agent, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    try:
        response = build_router().completion(**kwargs)
        return _extract_text(response)
    except Exception as exc:
        logger.warning(
            "llm_client_error",
            extra={"agent": agent, "error": str(exc), "trace_id": trace_id},
        )
        raise LLMRequestError(f"{agent}: {exc}") from exc