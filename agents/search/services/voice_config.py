import os
from typing import Dict

class VoiceConfig:
    STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en-US")
    STT_TIMEOUT = 5
    STT_PHRASE_TIME_LIMIT = 10
    TTS_RATE = 150
    TTS_VOLUME = 0.9
    TTS_VOICE_INDEX = None
    USE_GOOGLE_STT = True
    USE_PYTTSX3_TTS = True

    @classmethod
    def get_config(cls) -> Dict:
        return {
            "stt_language": cls.STT_LANGUAGE,
            "stt_timeout": cls.STT_TIMEOUT,
            "stt_phrase_time_limit": cls.STT_PHRASE_TIME_LIMIT,
            "tts_rate": cls.TTS_RATE,
            "tts_volume": cls.TTS_VOLUME,
            "tts_voice_index": cls.TTS_VOICE_INDEX,
        }
