import os
import requests
from utils.logger import get_logger

logger = get_logger(__name__)

# LM Studio exposes an OpenAI-compatible endpoint; same host/model convention
# as course_ingestion and planner agents.
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://host.docker.internal:1234/v1/chat/completions")
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
    """Call the LM Studio (Qwen / any local model) via its OpenAI-compatible API
    and return the assistant reply.

    Uses the same request pattern as `agents/course_ingestion/enrichment/llm_enricher.py`.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    env_url = os.getenv("LM_STUDIO_URL")
    if env_url:
        fallback_urls = [env_url]
    else:
        fallback_urls = [url, "http://host.docker.internal:1234/v1/chat/completions", "http://127.0.0.1:1234/v1/chat/completions"]
    seen = set()
    ordered_urls = []
    for candidate in fallback_urls:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered_urls.append(candidate)

    last_error = None
    for candidate_url in ordered_urls:
        try:
            response = requests.post(
                candidate_url,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 2000,
                },
                timeout=(2, LM_STUDIO_TIMEOUT_SECONDS)
            )
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]

            last_error = f"status={response.status_code}"
            logger.warning("lm_studio_error", extra={"status": response.status_code, "body": response.text[:200], "url": candidate_url})
        except Exception as exc:
            last_error = str(exc)
            logger.warning("lm_studio_exception", extra={"error": str(exc), "url": candidate_url})

    logger.warning("lm_unavailable", extra={"error": last_error})
    return ""
