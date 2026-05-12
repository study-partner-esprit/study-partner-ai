"""
Google Gemini LLM Client.
Uses google-genai SDK for chat completions.
Includes graceful handling of quota errors (429/RESOURCE_EXHAUSTED).
"""
import os
import time
import logging
from typing import Optional
from google import genai
from src.config.settings import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Constants for retry logic
RETRY_WAIT_SECONDS = 15
MAX_RETRIES = 1


class GeminiClient:
    """Google Gemini API client using google-genai SDK."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model: Optional Gemini model to use. If None, auto-selects best available model.
        
        Raises:
            ValueError: If API key is not provided.
            ImportError: If google-genai is not installed.
        """
        key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        
        if not key:
            logger.error("GEMINI_API_KEY not found in environment variables.")
            raise ValueError(
                "GEMINI_API_KEY not provided. Set via environment variable:\n"
                "  export GEMINI_API_KEY='your_key_here'"
            )
        
        try:
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

            # Query available models and select the best one available
            models = self.client.models.list()
            available_models = [m.name for m in models]
            print("Available Gemini models:", available_models)

            if model:
                self.model = model
            elif "gemini-1.5-flash-latest" in available_models:
                self.model = "gemini-1.5-flash-latest"
            elif "gemini-1.5-flash" in available_models:
                self.model = "gemini-1.5-flash"
            elif available_models:
                self.model = available_models[0]
            else:
                self.model = "gemini-1.5-flash"
            print(f"Using Gemini model: {self.model}")
            logger.info(f"✓ Gemini client initialized (model: {self.model})")
        except ImportError:
            logger.error("google-genai not installed. Install with: pip install google-genai")
            raise ImportError("google-genai package required. Install: pip install google-genai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def _is_quota_error(self, error: Exception) -> bool:
        """Check if error is a quota/rate limit error (429 or RESOURCE_EXHAUSTED)."""
        error_str = str(error).upper()
        return "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
    
    def _call_api_with_retry(self, prompt: str, max_tokens: int, temperature: float = 0.3) -> Optional[str]:
        """
        Make API call with retry logic for quota errors.
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum output tokens
            temperature: Model temperature
        
        Returns:
            API response text or None if quota exceeded after retry
        """
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                    }
                )
                
                if not response:
                    raise RuntimeError("Gemini returned empty response")
                
                text_result = response.text if hasattr(response, "text") else str(response)
                if not text_result:
                    raise RuntimeError("Gemini returned empty response")
                
                logger.debug(f"Gemini response: {text_result[:100]}...")
                return text_result
            
            except Exception as e:
                error_str = str(e)
                
                if self._is_quota_error(e):
                    logger.warning(f"Quota error (429/RESOURCE_EXHAUSTED) on attempt {attempt + 1}: {error_str}")
                    
                    if attempt < MAX_RETRIES:
                        logger.info(f"Waiting {RETRY_WAIT_SECONDS}s before retry...")
                        time.sleep(RETRY_WAIT_SECONDS)
                        logger.info("Retrying API call...")
                    else:
                        logger.error(f"Quota error persisted after {MAX_RETRIES + 1} attempts. Returning graceful message.")
                        return None
                else:
                    logger.error(f"Gemini API call failed (attempt {attempt + 1}): {error_str}")
                    raise RuntimeError(f"Gemini generation failed: {error_str}")
        
        return None
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
        """
        Generate text using Gemini API with quota error handling.
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum output tokens (default 150 to reduce API usage)
            temperature: Model temperature (default 0.3)
        
        Returns:
            Generated text, or user-friendly message if quota exceeded
        """
        result = self._call_api_with_retry(prompt, max_tokens, temperature)
        
        if result is None:
            return "⏱️ Quota exceeded. Please wait and try again."
        
        return result.strip()
    
    def chat(self, messages: list[dict], max_tokens: int = 150, temperature: float = 0.3) -> str:
        """
        Chat interface for multi-turn conversations with quota error handling.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum output tokens
            temperature: Model temperature
        
        Returns:
            Generated response, or user-friendly message if quota exceeded
        """
        try:
            prompt_text = ""
            for msg in messages:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                prompt_text += f"{role}: {content}\n\n"
            
            return self.generate(prompt_text, max_tokens, temperature)
        
        except Exception as e:
            logger.error(f"Gemini chat failed: {e}")
            raise RuntimeError(f"Gemini chat failed: {e}")
    
    def _validate_question(self, question: str, key_concepts: list[str] = None) -> bool:
        """
        Validate that a generated question meets quality criteria.
        
        Args:
            question: The question text to validate
            key_concepts: Optional list of key concepts that should be included
        
        Returns:
            True if valid, False if invalid
        """
        if not question:
            return False
        
        # Strip whitespace
        question = question.strip()
        
        # Check length (at least 10 words)
        words = question.split()
        if len(words) < 10:
            return False
        
        # Check ends with question mark
        if not question.endswith('?'):
            return False
        
        # Check doesn't start with incomplete phrases
        incomplete_starts = ["Considering", "Given", "Based on", "The", "A", "An", "In"]
        first_word = words[0] if words else ""
        if first_word in incomplete_starts and len(words) < 15:  # Allow if very long
            return False
        
        # Check for obvious redundant repeats like "process process" or "concept concept"
        lowered_words = [w.lower().strip(".,!?;") for w in words]
        for first, second in zip(lowered_words, lowered_words[1:]):
            if first == second:
                return False
        
        # Check includes at least one key concept if provided
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
        attempt_number: int = 1
    ) -> str:
        """
        Generate a validated Socratic question with quality checks and template fallback.
        
        Args:
            prompt: The prompt to send to Gemini
            max_tokens: Maximum output tokens
            temperature: Model temperature
            depth_level: Question depth level ("what", "why", "how")
            key_concepts: List of key concepts for template fallback
            task_title: Task title for template context
            task_details: Task details for richer context in templates
            attempt_number: Attempt number for template rotation
        
        Returns:
            Validated question, or template-based fallback if validation fails
        """
        from src.evaluator.prompts import generate_template_question
        
        key_concepts = key_concepts or []
        
        # Try up to 2 times to get a valid question from Gemini
        for attempt in range(2):
            result = self.generate(prompt, max_tokens, temperature)
            
            if result and self._validate_question(result, key_concepts):
                return result
            
            logger.warning(f"Question validation failed on attempt {attempt + 1}: '{result[:50]}...'")
        
        # Fallback to template-based question if Gemini fails
        logger.warning("Gemini question generation failed validation, using template fallback")
        return generate_template_question(
            depth_level=depth_level,
            key_concepts=key_concepts,
            task_title=task_title,
            task_details=task_details,
            attempt_number=attempt_number
        )


def get_gemini_client(api_key: Optional[str] = None, model: Optional[str] = None) -> GeminiClient:
    """
    Factory function to create Gemini client.
    
    Args:
        api_key: Optional API key override
        model: Optional model name. If None, auto-select best available model.
    
    Returns:
        GeminiClient instance
    """
    return GeminiClient(api_key=api_key, model=model)

