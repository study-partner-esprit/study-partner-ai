"""
Google Gemini LLM Client with LM Studio (Qwen) fallback.
Uses google-genai SDK for chat completions.
Includes graceful handling of quota errors (429/RESOURCE_EXHAUSTED).
"""
import os
import time
import logging
import requests
from typing import Optional
from google import genai
from src.config.settings import (
    GEMINI_API_KEY,
    LM_STUDIO_URL,
    LM_STUDIO_MODEL,
    LM_STUDIO_TIMEOUT,
    LM_STUDIO_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# Constants for retry logic
RETRY_WAIT_SECONDS = 15
MAX_RETRIES = 1


class QwenClient:
    """LM Studio (Qwen) client for local LLM inference fallback."""
    
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or LM_STUDIO_URL
        self.model = model or LM_STUDIO_MODEL
        self.timeout = LM_STUDIO_TIMEOUT
        self.temperature = LM_STUDIO_TEMPERATURE
        logger.info(f"✓ Qwen client initialized (model: {self.model}, url: {self.base_url})")
    
    def _call_api(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Call LM Studio API."""
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Qwen API call failed: {e}")
            return None
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
        """Generate text using Qwen via LM Studio."""
        result = self._call_api(prompt, max_tokens, temperature)
        if result is None:
            return "⏱️ Local model unavailable. Please check LM Studio."
        return result.strip()
    
    def chat(self, messages: list[dict], max_tokens: int = 150, temperature: float = 0.3) -> str:
        """Chat interface for multi-turn conversations."""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Qwen chat failed: {e}")
            raise RuntimeError(f"Qwen chat failed: {e}")
    
    def _validate_question(self, question: str, key_concepts: list[str] = None) -> bool:
        """Validate that a generated question meets quality criteria."""
        if not question:
            return False
        question = question.strip()
        words = question.split()
        if len(words) < 10:
            return False
        if not question.endswith('?'):
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
        attempt_number: int = 1
    ) -> str:
        """Generate a validated Socratic question with quality checks and template fallback."""
        from src.evaluator.prompts import generate_template_question
        
        key_concepts = key_concepts or []
        
        for attempt in range(2):
            result = self.generate(prompt, max_tokens, temperature)
            if result and self._validate_question(result, key_concepts):
                return result
            logger.warning(f"Qwen question validation failed on attempt {attempt + 1}: '{result[:50]}...'")
        
        logger.warning("Qwen question generation failed validation, using template fallback")
        return generate_template_question(
            depth_level=depth_level,
            key_concepts=key_concepts,
            task_title=task_title,
            task_details=task_details,
            attempt_number=attempt_number
        )


class GeminiClient:
    """Google Gemini API client using google-genai SDK with Qwen fallback."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, use_fallback: bool = True):
        """
        Initialize Gemini client with optional Qwen fallback.
        
        Args:
            api_key: Google Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model: Optional Gemini model to use. If None, auto-selects best available model.
            use_fallback: If True, initializes QwenClient as fallback for quota/API errors.
        
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
        
        # Initialize fallback client
        self.fallback = None
        if use_fallback:
            try:
                self.fallback = QwenClient()
                logger.info("✓ Qwen fallback client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Qwen fallback: {e}")
    
    def _is_quota_error(self, error: Exception) -> bool:
        """Check if error is a quota/rate limit error (429 or RESOURCE_EXHAUSTED)."""
        error_str = str(error).upper()
        return "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
    
    def _call_gemini_api(self, prompt: str, max_tokens: int, temperature: float = 0.3) -> Optional[str]:
        """
        Make API call to Gemini with retry logic for quota errors.
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
                        logger.error(f"Quota error persisted after {MAX_RETRIES + 1} attempts.")
                        return None
                else:
                    logger.error(f"Gemini API call failed (attempt {attempt + 1}): {error_str}")
                    raise RuntimeError(f"Gemini generation failed: {error_str}")
        
        return None
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
        """
        Generate text using Gemini API with Qwen fallback on quota/API errors.
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum output tokens
            temperature: Model temperature
        
        Returns:
            Generated text from Gemini, Qwen fallback, or error message
        """
        # Try Gemini first
        result = self._call_gemini_api(prompt, max_tokens, temperature)
        
        if result is not None:
            return result.strip()
        
        # Fallback to Qwen if available
        if self.fallback is not None:
            logger.info("Gemini failed/quota exceeded, falling back to Qwen")
            return self.fallback.generate(prompt, max_tokens, temperature)
        
        return "⏱️ Quota exceeded. Please wait and try again."
    
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

