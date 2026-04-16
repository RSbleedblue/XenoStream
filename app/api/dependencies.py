from fastapi import Request

from app.services.stt_engine import STTEngine
from app.services.tts_service import TTSService


def get_tts_service(request: Request) -> TTSService:
    return request.app.state.tts_service


def get_stt_engine(request: Request) -> STTEngine:
    return request.app.state.stt_engine
