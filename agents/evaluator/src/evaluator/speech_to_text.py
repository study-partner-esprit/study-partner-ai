"""
Whisper audio transcription module for student speech input.
Uses open-source Whisper model (no API costs).
"""

import logging
from pathlib import Path
from typing import Optional

try:
    import whisper
except ImportError:
    raise ImportError(
        "Whisper not installed. Install with: pip install openai-whisper"
    )

from src.config.settings import WHISPER_MODEL

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Transcribes audio files to text using OpenAI Whisper."""

    def __init__(self, model_name: str = WHISPER_MODEL):
        """
        Initialize Whisper transcriber.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
                       Default: base (good for most use cases)
        """
        self.model_name = model_name
        self._model: Optional[object] = None

    def _load_model(self) -> object:
        """Load Whisper model (cached after first load)."""
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self._model = whisper.load_model(self.model_name)
            logger.info(f"Whisper model loaded: {self.model_name}")
        return self._model

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)

        Returns:
            Transcribed text

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If transcription fails
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            logger.info(f"Transcribing audio: {audio_path}")
            model = self._load_model()

            # Transcribe with language auto-detection
            result = model.transcribe(
                str(audio_file),
                language=None,  # Auto-detect language
                fp16=False,  # Use float32 for compatibility
            )

            text = result["text"].strip()
            logger.info(f"Transcription complete: {len(text)} characters")

            return text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}")


# Global transcriber instance (lazy loaded)
_transcriber_instance: Optional[WhisperTranscriber] = None


def get_transcriber(model_name: str = WHISPER_MODEL) -> WhisperTranscriber:
    """
    Get or create global transcriber instance.

    Args:
        model_name: Whisper model size

    Returns:
        WhisperTranscriber instance
    """
    global _transcriber_instance

    if _transcriber_instance is None:
        _transcriber_instance = WhisperTranscriber(model_name)

    return _transcriber_instance


def transcribe_audio(audio_path: str) -> str:
    """
    Convenience function to transcribe audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Transcribed text
    """
    transcriber = get_transcriber()
    return transcriber.transcribe_audio(audio_path)
