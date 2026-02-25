import os
import requests
from utils.logger import get_logger

logger = get_logger(__name__)

# LM Studio exposes an OpenAI-compatible endpoint; same host/model convention
# as course_ingestion and planner agents.
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "lm-studio")

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

    try:
        response = requests.post(
            url,
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1024,
            },
            timeout=150,
        )
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            logger.warning("lm_studio_error", extra={"status": response.status_code, "body": response.text[:200]})
            return f"Error: LLM returned status {response.status_code}"
    except Exception as exc:
        logger.warning("lm_studio_exception", extra={"error": str(exc)})
        return f"Error LLM: {exc}"
