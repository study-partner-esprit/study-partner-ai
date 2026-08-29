import os
from utils.llm_client import LLMRequestError, MissingMockResponderError, ask
from utils.logger import get_logger

logger = get_logger(__name__)

# Legacy LM Studio defaults kept for backward-compatible imports; routing now
# goes through the shared client (S-MIG-01) → litellm/config.yaml (`search`).
LM_STUDIO_URL = os.getenv(
    "LM_STUDIO_URL", "http://host.docker.internal:1234/v1/chat/completions"
)
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "lm-studio")
LM_STUDIO_TIMEOUT_SECONDS = float(os.getenv("LM_STUDIO_TIMEOUT_SECONDS", "8"))

SYSTEM_PROMPT = (
    "You are a knowledgeable AI assistant. "
    "Answer questions clearly and concisely based only on the provided source material. "
    "Always respond in English."
)


def ask_llm(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    model: str = LM_STUDIO_MODEL,
    url: str = LM_STUDIO_URL,
) -> str:
    """Ask the `search` LLM through the shared LiteLLM client and return the
    assistant reply.

    `model`/`url` are accepted for backward compatibility and ignored. When no
    provider key is configured (or the call fails after router retries) an
    empty string is returned so the caller can degrade as before.
    """
    del model, url  # S-MIG-01: routing is owned by litellm/config.yaml
    try:
        return ask("search", system_prompt, prompt)
    except (LLMRequestError, MissingMockResponderError) as exc:
        logger.warning("search_llm_unavailable", extra={"error": str(exc)})
        return ""