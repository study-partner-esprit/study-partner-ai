"""LLM helpers for search agent — uses LM Studio (Qwen) via OpenAI-compatible API."""

from .llm import ask_llm, LM_STUDIO_URL, LM_STUDIO_MODEL

__all__ = ["ask_llm", "LM_STUDIO_URL", "LM_STUDIO_MODEL"]
