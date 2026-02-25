"""Backward compatibility small helpers mirroring the old `voice.py`.
"""
from .voice_service import get_voice_service, speak_answer


def get_voice_question():
    service = get_voice_service()
    return service.listen_for_question()


def speak_answer_text(text: str):
    return speak_answer(text)
