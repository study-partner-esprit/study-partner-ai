"""
Evaluator LLM client, routed through the shared LiteLLM client (S-MIG-01).
The `evaluator` model and its fallback chain are configured in
litellm/config.yaml; the local GeminiClient/QwenClient implementations are
retired. Quota/retry handling now lives in the LiteLLM router.
"""
import logging
from typing import Optional

from utils.llm_client import LLMRequestError, MissingMockResponderError, agent_config, ask

logger = logging.getLogger(__name__)

# Constants kept for backward compatibility (retry handling moved to the router).
RETRY_WAIT_SECONDS = 15
MAX_RETRIES = 1

# Short persona; the actual prompts already contain the task instructions.
EVALUATOR_SYSTEM_PROMPT = (
    "You are an expert Socratic tutor. Ask probing, single-concept questions and "
    "evaluate student understanding. Answer in terse, focused text."
)


class GeminiClient:
    """Evaluator LLM client backed by the shared `evaluator` model group.

    Keeps the historical name and public interface so consumers (EvaluatorAgent
    and standalone scripts) are unaffected by the migration.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, use_fallback: bool = True):
        # S-MIG-01: keys/models/routing are owned by litellm/config.yaml; the
        # constructor arguments are accepted for backward compatibility only.
        del api_key, use_fallback
        try:
            self.model = model or agent_config("evaluator").model
        except Exception:
            self.model = model or "evaluator"
        logger.info("✓ Evaluator LLM client initialized (model: %s)", self.model)

    @staticmethod
    def _generate_text(prompt: str, max_tokens: int, temperature: float) -> str:
        """Single round-trip through the shared `evaluator` model group."""
        try:
            return ask(
                "evaluator",
                EVALUATOR_SYSTEM_PROMPT,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ).strip()
        except (LLMRequestError, MissingMockResponderError) as e:
            logger.warning("Evaluator LLM unavailable: %s", e)
            return "⏱️ LLM unavailable. Please wait and try again."

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
        """Generate text using the shared `evaluator` model group."""
        return self._generate_text(prompt, max_tokens, temperature)

    def chat(self, messages: list[dict], max_tokens: int = 150, temperature: float = 0.3) -> str:
        """Chat interface for multi-turn conversations."""
        try:
            prompt_text = ""
            for msg in messages:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                prompt_text += f"{role}: {content}\n\n"
            return self.generate(prompt_text, max_tokens, temperature)
        except Exception as e:
            logger.error("Evaluator LLM chat failed: %s", e)
            raise RuntimeError(f"Evaluator LLM chat failed: {e}")

    def _validate_question(self, question: str, key_concepts: list[str] = None) -> bool:
        """Validate that a generated question meets quality criteria."""
        if not question:
            return False

        question = question.strip()
        words = question.split()
        if len(words) < 10:
            return False

        if not question.endswith("?"):
            return False

        incomplete_starts = ["Considering", "Given", "Based on", "The", "A", "An", "In"]
        first_word = words[0] if words else ""
        if first_word in incomplete_starts and len(words) < 15:
            return False

        lowered_words = [w.lower().strip(".,!?;") for w in words]
        for first, second in zip(lowered_words, lowered_words[1:]):
            if first == second:
                return False

        if key_concepts:
            question_lower = question.lower()
            has_concept = any(concept.lower() in question_lower for concept in key_concepts)
            if not has_concept:
                return False

        return True

    def generate_question(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.3,
        depth_level: str = "what",
        key_concepts: list[str] = None,
        task_title: str = "",
        task_details: str = "",
        attempt_number: int = 1,
    ) -> str:
        """Generate a validated Socratic question with quality checks and template fallback."""
        from agents.evaluator.prompts import generate_template_question

        key_concepts = key_concepts or []

        for attempt in range(2):
            result = self.generate(prompt, max_tokens, temperature)
            if result and self._validate_question(result, key_concepts):
                return result
            logger.warning(f"Question validation failed on attempt {attempt + 1}: '{result[:50]}...'")

        logger.warning("Question generation failed validation, using template fallback")
        return generate_template_question(
            depth_level=depth_level,
            key_concepts=key_concepts,
            task_title=task_title,
            task_details=task_details,
            attempt_number=attempt_number,
        )


def get_gemini_client(api_key: Optional[str] = None, model: Optional[str] = None) -> GeminiClient:
    """Factory function creating an evaluator GeminiClient (kept for compatibility)."""
    return GeminiClient(api_key=api_key, model=model)