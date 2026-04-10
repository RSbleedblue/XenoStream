from fastapi import Request

from app.services.tts_service import TTSService


def get_tts_service(request: Request) -> TTSService:
    return request.app.state.tts_service
