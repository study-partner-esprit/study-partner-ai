import speech_recognition as sr
import pyttsx3
import threading
from typing import Optional, Callable
from .voice_config import VoiceConfig
import logging

logger = logging.getLogger(__name__)


class VoiceService:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = None
        self._init_tts()

    def _init_tts(self):
        try:
            self.tts_engine = pyttsx3.init()
            if VoiceConfig.TTS_RATE:
                self.tts_engine.setProperty("rate", VoiceConfig.TTS_RATE)
            if VoiceConfig.TTS_VOLUME:
                self.tts_engine.setProperty("volume", VoiceConfig.TTS_VOLUME)
        except Exception as e:
            logger.error("Failed to init TTS: %s", e)
            self.tts_engine = None

    def listen_for_question(
        self,
        timeout: Optional[int] = None,
        phrase_time_limit: Optional[int] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        timeout = timeout or VoiceConfig.STT_TIMEOUT
        phrase_time_limit = phrase_time_limit or VoiceConfig.STT_PHRASE_TIME_LIMIT
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
            if VoiceConfig.USE_GOOGLE_STT:
                text = self.recognizer.recognize_google(
                    audio, language=VoiceConfig.STT_LANGUAGE
                )
            else:
                text = self.recognizer.recognize_sphinx(audio)
            text = text.strip()
            if callback:
                callback(text)
            return text
        except Exception:
            return ""

    def speak_text(
        self,
        text: str,
        async_mode: bool = False,
        callback: Optional[Callable[[], None]] = None,
    ) -> bool:
        if not text or not text.strip():
            return False
        if not self.tts_engine:
            return False

        def _speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                if callback:
                    callback()
            except Exception as e:
                logger.error("TTS error: %s", e)

        if async_mode:
            thread = threading.Thread(target=_speak, daemon=True)
            thread.start()
            return True
        else:
            _speak()
            return True


_voice_service_instance: Optional[VoiceService] = None


def get_voice_service() -> VoiceService:
    global _voice_service_instance
    if _voice_service_instance is None:
        _voice_service_instance = VoiceService()
    return _voice_service_instance


def speak_answer(text: str, async_mode: bool = False) -> bool:
    return get_voice_service().speak_text(text, async_mode=async_mode)
